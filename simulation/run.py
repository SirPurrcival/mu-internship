######################
## Import libraries ##
######################
import numpy as np
import nest
from functions import Network, raster, rate, get_irregularity, get_synchrony, get_firing_rate, prep_spikes, create_spectrogram
import time
import pickle
import pandas as pd

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
nest.local_num_threads = 32 ## adapt if necessary
nest.print_time = True if params['verbose'] == True else False
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

for i in range(len(params['layer_type'])):
    network.addpop('iaf_psc_exp', params['layer_type'][i] , int(num_neurons[i]), params['cell_params'], label=params['label'][i], record=True, nrec=int(params['R_scale']* num_neurons[i]))

###############################################################################
## Add background stimulation
for i in range(len(ext_rates)):
    network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rates[i], 'start': 0, 'stop': params['sim_time']}, 
                            target=params['layer_type'][i], 
                            c_specs={'rule': 'all_to_all'},
                            s_specs={'weight': params['ext_weights'][i],
                                     'delay': 1.5})

###############################################################################
## Connect the network
network.connect_all(params['layer_type'], connections, params['syn_type'], synaptic_strength)

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
network.connect(network.thalamic_population['thalamic_input'], pops['L4_E'], 
                conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0983 * len(network.get_pops()['L4_E']))}, 
                syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})
network.connect(network.thalamic_population['thalamic_input'], pops['L4_I'], 
                conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0619 * len(network.get_pops()['L4_I']))}, 
                syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})
network.connect(network.thalamic_population['thalamic_input'], pops['L6_E'], 
                conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0512 * len(network.get_pops()['L6_E']))}, 
                syn_specs={'weight': params['ext_weights'][0], 'delay': 1.5})
network.connect(network.thalamic_population['thalamic_input'], pops['L6_I'], 
                conn_specs={'rule': 'fixed_indegree', 'indegree': int(0.0196 * len(network.get_pops()['L6_I']))}, 
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
    ## Fetch data
    mmdata, spikes = network.get_data()
    
    ###########################################################################
    ## Do some data preparation for plotting
    ## Find out where to split
    index = np.insert(np.cumsum(network.get_nrec()), 0, 0)
    
    ## Split spikes into subarrays per population
    spike_data = [prep_spikes(spikes)[index[i]:index[i+1]] for i in range(len(index)-1)]
    
    ###########################################################################
    ## Calculate the average membrane potential across the whole network
    def average_membrane_voltage(data):
        df = pd.DataFrame(data)
        averaged_data = df.groupby('time_ms')['V_m'].mean().to_dict().values()
        #.to_dict()
        return np.array(list(averaged_data), dtype='float')
    
    vm_avg = average_membrane_voltage(mmdata)


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
        raster(spike_data, vm_avg, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), suffix=f"{str(int(params['th_in'])):0>4}")

        #######################################################################
        ## Display the average firing rate in Hz
        rate(spikes, params['rec_start'], params['rec_stop'])
        
        if params['verbose'] and rank == 0:
            print(f"Time required for graphing grid: {time.time() - st}")
        
        #######################################################################
        ## Plot spectrogram
        
        ## Max and min frequency in Hz
        f_min = 0
        f_max = 150
        
        ## Sampling rate in Hz
        fs = 1000 / params['resolution']
        num_timesteps = int((params['rec_start'] - params['rec_stop']) * (1000 / params['resolution']))
        
        create_spectrogram(vm_avg, fs, params['rec_start'], params['rec_stop'], f_min, f_max) 
    
    ###########################################################################
    ## Calculate measures 
    irregularity = [get_irregularity(population) for population in spike_data]
    firing_rate  = [get_firing_rate(population, params['rec_start'], params['rec_stop']) for population in spike_data]
    synchrony    = [get_synchrony(population, params['rec_start'], params['rec_stop']) for population in spike_data]
    
    ###########################################################################
    ## Write results to file
    with open("sim_results", 'wb') as f:
        data = (irregularity, synchrony, firing_rate)
        pickle.dump(data, f)
