######################
## Import libraries ##
######################
import numpy as np
import nest
from functions import Network, raster, rate, get_irregularity, get_synchrony, get_firing_rate, create_spectrogram, spectral_density
import time
import pickle
import pandas as pd

## Disable this if you run any kind of opt_ script
# from setup import setup
# setup()

###############################################################################
## Load config created by setup().
## It is done this way for optimization procedures that change parameter values
with open("params", 'rb') as f:
   params = pickle.load( f)

## Get the rank of the current process
rank = nest.Rank()

if params['verbose'] and rank == 0:
    print("Begin setup")
    ## get the start time
    st = time.time()

########################
## Set NEST Variables ##
########################
nest.ResetKernel()
nest.local_num_threads = 1 ## adapt if necessary
## nest.print_time = True if params['verbose'] == True else False
nest.resolution = params['resolution']
nest.set_verbosity("M_WARNING")
nest.overwrite_files = True
## Path relative to working directory
nest.data_path = "data/"
## Some random prefix that is given to all files (i.e. the trial number)
nest.data_prefix = "sim_data_"

#####################
# Scale the network #
#####################

synaptic_strength = params['syn_strength'] * params['syn_scale']
connections       = params['connectivity'] * params['K_scale']
num_neurons       = params['num_neurons']  * params['N_scale']
ext_rates         = params['ext_nodes']    * params['ext_rate']

########################
## Create the network ##
########################

network = Network(params)

###############################################################################
## Populate the network
if params['verbose'] and rank == 0:
    print(f"Time required for setup: {time.time() - st}\nPreparing network...")

for i in range(len(params['pop_name'])):
    network.addpop('iaf_psc_exp', params['pop_name'][i] , int(num_neurons[i]), params['cell_params'][i], record=True, nrec=int(params['R_scale']* num_neurons[i]))

###############################################################################
## Add background stimulation
for i in range(len(ext_rates)):
    network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rates[i], 'start': 0, 'stop': params['sim_time']}, 
                            target=params['pop_name'][i], 
                            c_specs={'rule': 'all_to_all'},
                            s_specs={'weight': params['ext_weights'][i],
                                     'delay': 1.5})

###############################################################################
## Connect the network
network.connect_all(params['pop_name'], connections, params['syn_type'], synaptic_strength)

###############################################################################
## Add thalamic input
## TODO: Make this prettier / more compact
network.addpop('parrot_neuron', 'thalamic_input', 902)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': params['th_in'], 'start': params['th_start'], 'stop': params['th_stop']}, 
                        target='thalamic_input', 
                        c_specs={'rule': 'all_to_all'},
                        s_specs={'weight': params['ext_weights'][0],
                                  'delay': 1.5})

pops = network.get_pops()
network.connect(network.thalamic_population['thalamic_input'], pops['L1_E'], 
                conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0983 * len(network.get_pops()['L1_E']))}, 
                syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})
network.connect(network.thalamic_population['thalamic_input'], pops['L1_I'], 
                conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0619 * len(network.get_pops()['L1_I']))}, 
                syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})
network.connect(network.thalamic_population['thalamic_input'], pops['L2_E'], 
                conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0512 * len(network.get_pops()['L1_E']))}, 
                syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})
network.connect(network.thalamic_population['thalamic_input'], pops['L2_I'], 
                conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0196 * len(network.get_pops()['L1_I']))}, 
                syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})

if params['verbose'] and rank == 0:
    print(f"Setup done after {time.time() - st}s! Starting simulation...")

###############################################################################
## Simulation loop
network.simulate(params['sim_time'])

if params['verbose'] and rank == 0:
    print(f"Total time required for simulation: {time.time() - st}\nDone, fetching and preparing data...")

###############################################################################
## Data fetching and preparation
if rank == 0:
    ###########################################################################
    ## Create results dictionary
    results = {}
    
    ###########################################################################
    ## Fetch data
    mmdata, spikes = network.get_data()
    
    ###########################################################################
    ## Do some data preparation for plotting
    ## Find out where to split
    index = np.insert(np.cumsum(network.get_nrec()), 0, 0)
    
    ## Split spikes into subarrays per population
    spike_data = [network.prep_spikes(spikes)[index[i]:index[i+1]] for i in range(len(index)-1)]
    
    ## Check for silent populations
    run_borked = False
    for population, name in zip(spike_data, params['pop_name']):
        ## Print the number of silent neurons
        silent_neurons = sum([1 if len(x) == 0 else 0 for x in population])
        print(f"Number of silent neurons in {name}: {silent_neurons}")
        if silent_neurons > 0:
            run_borked = True
            results['ISI_mean'] = -1
            results['ISI_std']  = -1
            results['CV']       = -1
            with open("sim_results", 'wb') as f:
                pickle.dump(results, f)
            raise Exception("Run borked, silent neurons")
            
    
    ## split vm into the two networks
    mm_1, mm_2 = network.split_mmdata(mmdata)
    
    
    ###########################################################################
    ## Calculate the average membrane potential across the whole network
    def average_membrane_voltage(data):
        df = pd.DataFrame(data)
        averaged_data = df.groupby('time_ms')['V_m'].mean().to_dict().values()
        return np.array(list(averaged_data), dtype='float')
    
    vm_avg_1 = average_membrane_voltage(mm_1)
    vm_avg_2 = average_membrane_voltage(mm_2)

    if params['verbose'] and rank == 0:
        print(f"Done. Time taken for data gathering and preparation: {time.time() - st}")
    
    
    ###########################################################################
    ## Plotting and presentation
    ## If we're not optimizing, plot
    if not params['opt_run']:
        
        print("Graphing spikes...")
        ## Define colors used in the raster plot per neuron population based on label
        colors = ["crimson" if x.split("_")[1] == "I" else "tab:blue" for x in network.get_pops().keys()]
        
        #######################################################################
        ## Plot spike data
        raster(spike_data[:2], vm_avg_1, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), prefix="N1_", suffix=f"{str(int(params['th_in'])):0>4}")
        raster(spike_data[2:], vm_avg_2, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), prefix="N2_", suffix=f"{str(int(params['th_in'])):0>4}")
        raster(spike_data, np.average([vm_avg_1, vm_avg_2], axis=0), params['rec_start'], params['rec_stop'], colors, network.get_nrec(), prefix="N3_", suffix=f"{str(int(params['th_in'])):0>4}")

        #######################################################################
        ## Display the average firing rate in Hz
        rate(spike_data, params['rec_start'], params['rec_stop'])
        
        if params['verbose'] and rank == 0:
            print(f"Time required for graphing grid: {time.time() - st}")
        
        #######################################################################
        ## Plot spectrogram
        
        ## Max and min frequency in Hz
        f_min = 0
        f_max = 100
        
        ## Sampling rate in Hz
        fs = 1000 / params['resolution']
        num_timesteps = int((params['rec_stop'] - params['rec_start']) * (1000 / params['resolution']))
        
        create_spectrogram(vm_avg_1, fs, params['rec_start'], params['rec_stop'], f_min, f_max)
        create_spectrogram(vm_avg_2, fs, params['rec_start'], params['rec_stop'], f_min, f_max)
        
        #spectral_density(len(vm_avg_1), params['resolution']*1e-3, vm_avg_1, get_peak=True)
        #spectral_density(len(vm_avg_1), params['resolution']*1e-3, vm_avg_2, get_peak=True)
        
    ## Calculate the highest frequency in the data
    peaks = spectral_density(len(vm_avg_1), params['resolution']*1e-3, [vm_avg_1, vm_avg_2], plot=not params['opt_run'], get_peak=True)
    
    ## Check for synchronous behaviour
    for i in range(len(peaks)):
        if peaks[i] < 0:
            print(f"No synchronous behaviour in network {i+1}")
        else:
            print(f"Synchronous behaviour in network {i+1} with a peak at {peaks[i]} Hz")
            
    
    ## Difference in frequencies between the populations
    df = abs(peaks[0] - peaks[1])
        
        
    from scipy.signal import correlate, welch
       
    def analyze_spike_trains(spike_trains, resolution=0.1):
        """
        Parameters are chosen in such a way as to havea frequency resolution of 1Hz. 
        Time resolution scales with more available data, meaning higher frequency or 
        longer simulations as nperseg can be maximally as high as the number of data points.
        """
        
        ## Convert resolution to seconds
        resolution_sec = resolution / 1000.0
        
        ## Calculate autocorrelations per neuron
        autocorrelations = [correlate(spike_train, spike_train[::-1], mode='full') for spike_train in spike_trains]
        
        ## Calculate power spectral densities
        power_spectral_densities = [welch(autocorr, 
                                          fs=1 / resolution_sec, 
                                          nperseg=min(len(autocorr), int(1 / resolution_sec)), 
                                          nfft=int(1 / resolution_sec)) for autocorr in autocorrelations]
        
        ## Restrict the frequency range
        power_spectral_densities = [(freqs[valid_indices], powers[valid_indices]) for freqs, powers in power_spectral_densities \
                                    if len(valid_indices := np.where((freqs >= 5) & (freqs <= 100))[0]) > 0]
            
    
        ## Frequency bands of potential interest (mostly beta and gamma)
        frequency_bands = {
            'theta': (4 ,  8),
            'alpha': (8 , 12),
            'beta' : (13, 30),
            'gamma': (30, 80)
        }
        
        ## Compute the total power per neuron in each of the previously defined
        ## frequency bands. Probably needs to be normalized
        ## TODO: Normalize
        band_powers = {band: [] for band in frequency_bands}
    
        for power_spectrum in power_spectral_densities:
            freqs, powers = power_spectrum
            for band, (fmin, fmax) in frequency_bands.items():
                band_indices = np.where((freqs >= fmin) & (freqs <= fmax))[0]
                if len(band_indices) > 0:
                    band_power = np.sum(powers[band_indices])
                    band_powers[band].append(band_power)
                else:
                    band_powers[band].append(0)
        
        ## Compute measures of synchrony
        synchrony_measures = {}
    
        ## Peak coherence. Frequency with the highest power
        synchrony_measures['peak_coherence'] = [np.argmax(powers) for freqs, powers in power_spectral_densities]
    
        ## Total power
        synchrony_measures['total_power'] = [np.sum(powers) for _, powers in power_spectral_densities]
    
        gamma_power = np.mean(band_powers['gamma'])
        theta_power = np.mean(band_powers['theta'])
        synchrony_measures['gamma_to_theta_ratio'] = gamma_power / theta_power
        
        out_dict = {
            'autocorrelations'        : autocorrelations,
            'power spectral densities': power_spectral_densities,
            'band powers'             : band_powers,
            'synchrony measures'      : synchrony_measures
            }
        
        return  out_dict
            
    outcomes = []
    for data in spike_data:
        outcomes.append(analyze_spike_trains(data))
    
    if not params['opt_run']:
        import matplotlib.pyplot as plt
        for outcome in outcomes:
            for item in outcome['band powers'].values():
                plt.plot(item)
            plt.legend(outcome['band powers'].keys())
            plt.show()
        if params['verbose'] and rank == 0:
            print(f"Done. Final time: {time.time() - st}")
    
    ###########################################################################
    ## Calculate measures 
    # irregularity = [get_irregularity(population) for population in spike_data]
    # firing_rate  = [get_firing_rate(population, params['rec_start'], params['rec_stop']) for population in spike_data]
    # synchrony    = [get_synchrony(population, params['rec_start'], params['rec_stop']) for population in spike_data]
    
    
    def ISI(data):
        return np.concatenate([np.diff(d) for d in data])
    
    def calculate_CV(isi_data):
        isi_mean = np.mean(isi_data)
        isi_std = np.std(isi_data)
        cv = isi_std / isi_mean
        return cv
    
    ISI_data = [ISI(d) for d in spike_data]
    ISI_mean = [np.mean(d) for d in ISI_data]
    ISI_std  = [np.std(d) for d in ISI_data]
    ISI_CV   = [std / mean for std, mean in zip(ISI_std, ISI_mean)]
    
    results['ISI_mean'] = ISI_mean
    results['firing_rate'] = 1/(np.array(ISI_mean)/1000)
    results['ISI_std']  = ISI_std
    results['CV']       = ISI_CV
    
    ## Write to disk
    with open("sim_results", 'wb') as f:
        pickle.dump(results, f)
