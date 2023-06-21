######################
## Import libraries ##
######################
import numpy as np
import nest
from functions import Network, raster, rate, get_irregularity, get_synchrony, get_firing_rate, create_spectrogram, spectral_density, split_mmdata, compute_plv
import time
import pickle
import pandas as pd
import os

## Disable this if you run any kind of opt_ script
from setup import setup
setup()

###############################################################################
## Load config created by setup().
## It is done this way for optimization procedures that change parameter values
with open("params", 'rb') as f:
   params = pickle.load( f)

## Get the rank of the current process
rank = nest.Rank()

if params['verbose'] and rank == 0:
    print("Begin Setup")
    ## get the start time
    st = time.time()

########################
## Set NEST Variables ##
########################
nest.ResetKernel()
nest.local_num_threads = 4 ## adapt if necessary
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

####################
## Create network ##
####################

def create_network(params):
    network = Network(params)
    
    ###############################################################################
    ## Populate the network
    if params['verbose'] and rank == 0:
        print(f"Time required for setup: {time.time() - st}\nRunning Nest with {nest.NumProcesses()} workers")
    
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
    # network.addpop('parrot_neuron', 'thalamic_input', 902)
    # network.add_stimulation(source={'type': 'poisson_generator', 'rate': params['th_in'], 'start': params['th_start'], 'stop': params['th_stop']}, 
    #                         target='thalamic_input', 
    #                         c_specs={'rule': 'all_to_all'},
    #                         s_specs={'weight': params['ext_weights'][0],
    #                                   'delay': 1.5})
    
    # pops = network.get_pops()
    # network.connect(network.thalamic_population['thalamic_input'], pops['L1_E'], 
    #                 conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0983 * len(network.get_pops()['L1_E']))}, 
    #                 syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})
    # network.connect(network.thalamic_population['thalamic_input'], pops['L1_I'], 
    #                 conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0619 * len(network.get_pops()['L1_I']))}, 
    #                 syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})
    # network.connect(network.thalamic_population['thalamic_input'], pops['L2_E'], 
    #                 conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0512 * len(network.get_pops()['L1_E']))}, 
    #                 syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})
    # network.connect(network.thalamic_population['thalamic_input'], pops['L2_I'], 
    #                 conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0196 * len(network.get_pops()['L1_I']))}, 
    #                 syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})
    
    if params['verbose'] and rank == 0:
        print(f"Setup done after {time.time() - st}s! Starting simulation...")
    return network

second_net = params['second_net']

network1 = create_network(params)
if second_net:
    network2 = create_network(params)
        

###############################################################################
## Connect networks here

        


###############################################################################
## Simulation loop

nest.Simulate(params['sim_time'])
##network.simulate(params['sim_time'])

if params['verbose'] and rank == 0:
    print(f"Total time required for simulation: {time.time() - st}\nDone, fetching and preparing data...")

###########################################################################
## Fetch data
net1_tmp_mm_data, net1_spikes = network1.get_data()
net1_nrec = network1.get_nrec()

if second_net:
    net2_tmp_mm_data, net2_spikes = network2.get_data()
    net2_nrec = network2.get_nrec()

###########################################################################
## Do some data preparation for plotting
## Find out where to split
net1_indices = np.insert(np.cumsum(net1_nrec), 0, 0)

if second_net:
    net2_indices = np.insert(np.cumsum(net2_nrec), 0, 0)

## Split spikes into subarrays per population
net1_tmp_sr_data = [network1.prep_spikes(net1_spikes)[net1_indices[i]:net1_indices[i+1]] for i in range(len(net1_indices)-1)]

if second_net:
    net2_tmp_sr_data = [network2.prep_spikes(net2_spikes)[net2_indices[i]:net2_indices[i+1]] for i in range(len(net2_indices)-1)]

## Only save and load if we're using more than one worker
if nest.NumProcesses() > 1:
    with open(f"data/net1_tmp_sr_{rank}", 'wb') as f:
        pickle.dump(net1_tmp_sr_data, f)
        
    with open(f"data/net1_tmp_mm_{rank}", 'wb') as f:
        pickle.dump(net1_tmp_mm_data, f)
    if second_net:
        with open(f"data/net2_tmp_sr_{rank}", 'wb') as f:
            pickle.dump(net2_tmp_sr_data, f)
            
        with open(f"data/net2_tmp_mm_{rank}", 'wb') as f:
            pickle.dump(net2_tmp_mm_data, f)


## Wait until all processes have finished
nest.SyncProcesses()
if params['verbose'] and rank == 0:
    print("Done. Processing data")

###############################################################################
## Data fetching and preparation
if rank == 0:
    ## If more than one process is used we need to take care of merging data
    if nest.NumProcesses() > 1:
        ## Merge different ranks back together
        sr_filenames = [filename for filename in os.listdir('data/') if filename.startswith("tmp_sr_")]
        mm_filenames = [filename for filename in os.listdir('data/') if filename.startswith("sim_data_multimeter")]
        
        rank_data_sr = []
        rank_data_mm = []
        
        for fn in sr_filenames:
            with open("data/"+fn, 'rb') as f:
               rank_data_sr.append(pickle.load( f))
        
        def merge_spike_data(data_list):
            num_ranks = len(data_list)
            
            ## Data from rank 0
            for i in range(len(data_list[0])):
                ## index of population
                for j in range(len(data_list[0][i])):
                    data_list[0][i][j] = data_list[j%num_ranks][i][j-(j%num_ranks)]
            
            return data_list[0]
        
        if params['verbose'] and rank == 0:
            print(f"Time before V_m merge: {time.time() - st}")
        ## Merge mm data
        dataframes = []
        for file_path in mm_filenames:
            df = pd.read_csv("data/"+file_path, sep='\t', comment='#', header=3, names=["sender", "time", "V_m"])
            dataframes.append(df)
        
        merged_dataframe = pd.concat(dataframes)
        
        # Sort the merged data by time
        merged_dataframe.sort_values("time", inplace=True)
        
        if params['verbose'] and rank == 0:
            print(f"Time after V_m merge: {time.time() - st}")
    else:
        net1_spike_data = net1_tmp_sr_data
        net1_mm_data = net1_tmp_mm_data
        
        if second_net:
            net2_spike_data = net2_tmp_sr_data
            net2_mm_data = net2_tmp_mm_data
    
    ###########################################################################
    ## Create results dictionary
    results = {}
    
    ## Check for silent populations
    run_borked = False
    for population, name in zip(net1_spike_data, params['pop_name']):
        ## Print the number of silent neurons
        silent_neurons = sum([1 if len(x) == 0 else 0 for x in population])
        print(f"Number of silent neurons in {name}: {silent_neurons}")
        if silent_neurons > 0:
            run_borked = True
            results['ISI_mean']    = -1
            results['ISI_std']     = -1
            results['CV']          = -1
            results['firing_rate'] = -1
            with open("sim_results", 'wb') as f:
                pickle.dump(results, f)
            raise Exception(f"Run borked, silent neurons in rank {rank}")
        
            
    
    ## split vm into the two networks
    net1_mm_1, net1_mm_2 = split_mmdata(net1_mm_data, net1_nrec)
    if second_net:
        net2_mm_1, net2_mm_2 = split_mmdata(net2_mm_data, net2_nrec)
    
    ###########################################################################
    ## Calculate the average membrane potential across the whole network
    def average_membrane_voltage(data):
        df = pd.DataFrame(data)
        averaged_data = df.groupby('time_ms')['V_m'].mean().to_dict().values()
        return np.array(list(averaged_data), dtype='float')
    
    net1_vm_avg_1 = average_membrane_voltage(net1_mm_1)
    net1_vm_avg_2 = average_membrane_voltage(net1_mm_2)
    
    if second_net:
        net2_vm_avg_1 = average_membrane_voltage(net2_mm_1)
        net2_vm_avg_2 = average_membrane_voltage(net2_mm_2)
    
    if params['verbose'] and rank == 0:
        print(f"Done. Time taken for data gathering and preparation: {time.time() - st}")
    
    ###########################################################################
    ## Plotting and presentation
    ## If we're not optimizing, plot
    if not params['opt_run']:
        
        print("Graphing spikes...")
        ## Define colors used in the raster plot per neuron population based on label
        colors = ["crimson" if x.split("_")[1] == "I" else "tab:blue" for x in network1.get_pops().keys()]
        
        #######################################################################
        ## Plot spike data
        # raster(spikes[:2], vm_avg_1, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), prefix="N1_", suffix=f"{str(int(params['th_in'])):0>4}")
        # raster(spikes[2:], vm_avg_2, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), prefix="N2_", suffix=f"{str(int(params['th_in'])):0>4}", pops_to_plot=[1,2])
        raster(net1_spikes, np.average([net1_vm_avg_1, net1_vm_avg_2], axis=0), params['rec_start'], params['rec_stop'], colors, net1_nrec, prefix="N1_", suffix=f"{str(int(params['th_in'])):0>4}")
        
        if second_net:
            raster(net2_spikes, np.average([net2_vm_avg_1, net2_vm_avg_2], axis=0), params['rec_start'], params['rec_stop'], colors, net2_nrec, prefix="N2_", suffix=f"{str(int(params['th_in'])):0>4}")
        
        #######################################################################
        ## Display the average firing rate in Hz
        rate(net1_spike_data, params['rec_start'], params['rec_stop'])
        
        if second_net:
            rate(net2_spike_data, params['rec_start'], params['rec_stop'])
        
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
        
        create_spectrogram(net1_vm_avg_1, fs, params['rec_start'], params['rec_stop'], f_min, f_max)
        create_spectrogram(net1_vm_avg_2, fs, params['rec_start'], params['rec_stop'], f_min, f_max)
        
        #spectral_density(len(vm_avg_1), params['resolution']*1e-3, vm_avg_1, get_peak=True)
        #spectral_density(len(vm_avg_1), params['resolution']*1e-3, vm_avg_2, get_peak=True)
        
    ## Calculate the highest frequency in the data
    peaks = spectral_density(len(net1_vm_avg_1), params['resolution']*1e-3, [net1_vm_avg_1, net1_vm_avg_2], plot=not params['opt_run'], get_peak=True)
    
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
    for data in net1_spike_data:
        outcomes.append(analyze_spike_trains(data))
    
    # if not params['opt_run']:
    #     import matplotlib.pyplot as plt
    #     for outcome in outcomes:
    #         for item in outcome['band powers'].values():
    #             plt.plot(item)
    #         plt.legend(outcome['band powers'].keys())
    #         plt.show()
    #     if params['verbose'] and rank == 0:
    #         print(f"Done. Final time: {time.time() - st}")
    
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
    
    net1_ISI_data = [ISI(d) for d in net1_spike_data]
    net1_ISI_mean = [np.mean(d) for d in net1_ISI_data]
    net1_ISI_std  = [np.std(d) for d in net1_ISI_data]
    net1_ISI_CV   = [std / mean for std, mean in zip(net1_ISI_std, net1_ISI_mean)]
    
    plv = compute_plv(net1_spike_data[0], net1_spike_data[2], params['rec_start'], params['rec_stop'], bin_width=3)
    print("Phase-Locking Value (PLV):", plv)

    results['PLV']         = plv
    results['ISI_mean']    = np.array(net1_ISI_mean)
    results['firing_rate'] = 1/(np.array(net1_ISI_mean)/1000)
    results['ISI_std']     = np.array(net1_ISI_std)
    results['CV']          = np.array(net1_ISI_CV)
    
    ## Write to disk
    with open("sim_results", 'wb') as f:
        pickle.dump(results, f)
    