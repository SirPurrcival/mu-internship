######################
## Import libraries ##
######################
import numpy as np
import nest
from functions import Network, raster, rate, create_spectrogram, spectral_density, split_mmdata, compute_plv, analyze_spike_trains, ISI
import time
import pickle
import pandas as pd
import os

## Disable this if you run any kind of opt_ script
# from setup import setup
# setup()

###############################################################################
## Load config created by setup()
## It is done this way for optimization procedures that change parameter values

with open("params", 'rb') as f:
   params = pickle.load( f)

## Get the rank of the current process
rank = nest.Rank()

if params['verbose'] and rank == 0:
    print("Begin Setup")
    ## get the start time
    st = time.time()

seed = 13515

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
nest.rng_seed = seed
np.random.seed(seed)

#####################
# Scale the network #
#####################

## Doesnt do anything yet, ignore
synaptic_strength = params['syn_strength'] * params['syn_scale']
interlaminar      = params['interlaminar'] * params['K_scale']
intralaminar      = params['intralaminar'] * params['K_scale']
num_neurons       = params['num_neurons']  * params['N_scale']
ext_rates         = params['ext_nodes']    * params['ext_rate']

####################
## Create network ##
####################

def create_network(params, net):
    network = Network(params)
    
    ###############################################################################
    ## Populate the network
    if params['verbose'] and rank == 0:
        print(f"Time required for setup: {time.time() - st}\nRunning Nest with {nest.NumProcesses()} workers")
    
    for i in range(len(params['pop_name'])):
        network.addpop('iaf_psc_exp', params['pop_name'][i] , int(num_neurons[i]), params[f'cell_params_net{net}'][i], record=True, nrec=int(params['R_scale']* num_neurons[i]))
    
    ###############################################################################
    ## Add background stimulation
    for i in range(len(ext_rates)):
        network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rates[i], 'start': 0, 'stop': params['sim_time']}, 
                                target=params['pop_name'][i], 
                                c_specs={'rule': 'all_to_all'},
                                s_specs={'weight': params['ext_weights'][i],
                                         'delay': 1.5})
    
    ###############################################################################
    ## Connect the network. Do intralaminar first so that network will aways be the same regardless
    ## of interlaminar connections
    network.connect_all(params['pop_name'], intralaminar, params['syn_type'], synaptic_strength)
    network.connect_all(params['pop_name'], interlaminar, params['syn_type'], synaptic_strength)
    
    ###############################################################################
    ## Add thalamic input || NOT IMPLEMENTED || 
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

network1 = create_network(params, 1)
if second_net:
    network2 = create_network(params, 2)
   
###############################################################################
## Deep layer input for network 1 
   
stimulus = nest.Create('poisson_generator')
stimulus.rate = params['deep_input']
stimulus.start = params['rec_start']
stimulus.stop = params['rec_stop']

nest.Connect(stimulus, network1.get_pops()["L2_E"], {'rule': 'all_to_all'}, {'weight': params['ext_weights'][0],'delay': 1.5})


###############################################################################
## Connect networks here

if second_net:
    
    net1_pops = network1.get_pops()
    net2_pops = network2.get_pops()
    
    for popa, i in zip(network1.get_pops(), range(len(network1.get_pops()))):
        for popb, j in zip(network2.get_pops(), range(len(network2.get_pops()))):
            
            if params['syn_type'][i,j] == "E":
                receptor_type = 1
                w_min = 0.0
                w_max = np.inf
            else:
                receptor_type = 2
                w_min = np.NINF
                w_max = 0.0
                
            weight = nest.math.redraw(nest.random.normal(
                mean = params['syn_strength'][i,j],
                std=abs(params['syn_strength'][i,j]) * params['sd']),
                min=w_min,
                max=w_max)
            delay = nest.math.redraw(nest.random.normal(
                mean = 1.5 if params['syn_strength'][i,j] == "E" else 0.75,
                std=abs(1.5*0.5) if params['syn_strength'][i,j] == "E" else abs(0.75*0.5)),
                min=params['resolution'], 
                max=np.Inf)
            
            nest.Connect(net1_pops[popa], net2_pops[popb],
                         conn_spec = {'rule': 'fixed_indegree', 
                                      'indegree': int(params['net1_net2_connections'][i,j] * len(net1_pops[popa]))},
                         syn_spec = {"weight": weight, "delay": delay})#, "receptor_type": receptor_type})
            
            if params['syn_type'][j,i] == "E":
                receptor_type = 1
                w_min = 0.0
                w_max = np.inf
            else:
                receptor_type = 2
                w_min = np.NINF
                w_max = 0.0
                
            weight = nest.math.redraw(nest.random.normal(
                mean = params['syn_strength'][j,i],
                std=abs(params['syn_strength'][j,i]) * params['sd']),
                min=w_min,
                max=w_max)
            delay = nest.math.redraw(nest.random.normal(
                mean = 1.5 if params['syn_strength'][j,i] == "E" else 0.75,
                std=abs(1.5*0.5) if params['syn_strength'][j,i] == "E" else abs(0.75*0.5)),
                min=params['resolution'], 
                max=np.Inf)
            
            nest.Connect(net2_pops[popb], net1_pops[popa], 
                         conn_spec = {'rule': 'fixed_indegree', 
                         'indegree': int(params['net2_net1_connections'][j,i] * len(net2_pops[popb]))},
            syn_spec = {"weight": weight, "delay": delay})#, "receptor_type": receptor_type})


###############################################################################
## Simulation loop

nest.Simulate(params['sim_time'])
##network.simulate(params['sim_time'])

if params['verbose'] and rank == 0:
    print(f"Total time required for simulation: {time.time() - st}\nDone, fetching and preparing data...")

###########################################################################
## Fetch data
net1_mm_data, net1_spikes = network1.get_data()
net1_nrec = network1.get_nrec()

if second_net:
    net2_mm_data, net2_spikes = network2.get_data()
    net2_nrec = network2.get_nrec()

###########################################################################
## Do some data preparation for plotting
## Find out where to split
net1_indices = np.insert(np.cumsum(net1_nrec), 0, 0)

if second_net:
    net2_indices = np.insert(np.cumsum(net2_nrec), 0, 0)

## Only save and load if we're using more than one worker and recorded to memory
if nest.NumProcesses() > 1 and params['record_to'] == 'memory':
    with open(f"data/net1_spikes_{rank}", 'wb') as f:
        pickle.dump(net1_spikes, f)
        
    with open(f"data/net1_mm_{rank}", 'wb') as f:
        pickle.dump(net1_mm_data, f)
    if second_net:
        with open(f"data/net2_spikes_{rank}", 'wb') as f:
            pickle.dump(net2_spikes, f)
            
        with open(f"data/net2_mm_{rank}", 'wb') as f:
            pickle.dump(net2_mm_data, f)


## Wait until all processes have finished
nest.SyncProcesses()
if params['verbose'] and rank == 0:
    print("Done. Processing data")

###############################################################################
## Data fetching and preparation
if rank == 0:
    ## If more than one process is used we need to take care of merging data
    if nest.NumProcesses() > 1:
        if params['record_to'] != 'memory':
            ## Merge different ranks back together
            sr_filenames = [filename for filename in os.listdir('data/') if filename.startswith("sim_data_spike_recorder")]
            mm_filenames = [filename for filename in os.listdir('data/') if filename.startswith("sim_data_multimeter")]
            
            rank_data_sr = [pickle.load(open("data/"+fn, 'rb')) for fn in sr_filenames]
            rank_data_mm = [pickle.load(open("data/"+fn, 'rb')) for fn in mm_filenames]
                   
        
            if params['verbose'] and rank == 0:
                print(f"Time before V_m merge: {time.time() - st}")
                
            sr_data = [pd.read_csv("data/"+file_path, sep='\t', comment='#', header=3, names=["sender", "time", "V_m"]) for file_path in mm_filenames]
            mm_data = [pd.read_csv("data/"+file_path, sep='\t', comment='#', header=3, names=["sender", "time"]) for file_path in sr_filenames]
            
            merged_mm = pd.concat(mm_data)
            merged_sr = pd.concat(sr_data)
            
            # Sort the merged data by time
            net1_spikes.sort_values("time", inplace=True)
            merged_sr.sort_values("time", inplace=True)
        
            if params['verbose'] and rank == 0:
                print(f"Time after V_m merge: {time.time() - st}")
        else:
            pass
    else:
        pass
            
    ## Split spikes into subarrays per population
    net1_spike_data = [network1.prep_spikes(net1_spikes)[net1_indices[i]:net1_indices[i+1]] for i in range(len(net1_indices)-1)]

    if second_net:
        net2_spike_data = [network2.prep_spikes(net2_spikes)[net2_indices[i]:net2_indices[i+1]] for i in range(len(net2_indices)-1)]
    
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
            results['net1_ISI_mean']         = -1
            results['net1_ISI_std']          = -1
            results['net1_CV']               = -1
            results['net1_firing_rate']      = -1
            
            results['net2_ISI_mean']         = -1
            results['net2_ISI_std']          = -1
            results['net2_CV']               = -1
            results['net2_firing_rate']      = -1
            
            results['PLV_intracircuit_net1'] = -1
            results['PLV_intracircuit_net2'] = -1
            results['PLV_intercircuit']      = -1
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
    peaks_1 = spectral_density(len(net1_vm_avg_1), params['resolution']*1e-3, [net1_vm_avg_1, net1_vm_avg_2], plot=not params['opt_run'], get_peak=True)
    if second_net:
        peaks_2 = spectral_density(len(net2_vm_avg_1), params['resolution']*1e-3, [net2_vm_avg_1, net2_vm_avg_2], plot=not params['opt_run'], get_peak=True)
    
    ## Check for synchronous behaviour
    for i in range(len(peaks_1)):
        if peaks_1[i] < 0:
            print(f"No synchronous behaviour in network {i+1}")
        else:
            print(f"Synchronous behaviour layer {i+1} with a peak at {peaks_1[i]} Hz")
    
    if second_net:
        for i in range(len(peaks_2)):
            if peaks_2[i] < 0:
                print(f"No synchronous behaviour in network {i+1}")
            else:
                print(f"Synchronous behaviour layer {i+1} with a peak at {peaks_2[i]} Hz")
            
    
    ## Difference in frequencies between the populations
    df = abs(peaks_1[0] - peaks_1[1])
       
    
            
    outcomes = []
    for data in net1_spike_data:
        outcomes.append(analyze_spike_trains(data))
    

    net1_ISI_data = [ISI(d) for d in net1_spike_data]
    net1_ISI_mean = [np.mean(d) for d in net1_ISI_data]
    net1_ISI_std  = [np.std(d) for d in net1_ISI_data]
    net1_ISI_CV   = [std / mean for std, mean in zip(net1_ISI_std, net1_ISI_mean)]
    
    if second_net:
        net2_ISI_data = [ISI(d) for d in net2_spike_data]
        net2_ISI_mean = [np.mean(d) for d in net2_ISI_data]
        net2_ISI_std  = [np.std(d) for d in net2_ISI_data]
        net2_ISI_CV   = [std / mean for std, mean in zip(net2_ISI_std, net2_ISI_mean)]
    
        spk_net1_upper = np.sort(np.concatenate(net1_spike_data[0]))
        spk_net2_upper = np.sort(np.concatenate(net2_spike_data[0]))
        
        spk_net1_upper = np.sort(np.concatenate(net1_spike_data[0]))
        spk_net1_lower = np.sort(np.concatenate(net1_spike_data[2]))
        
        spk_net2_upper = np.sort(np.concatenate(net2_spike_data[0]))
        spk_net2_lower = np.sort(np.concatenate(net2_spike_data[2]))
        
        plv_intracircuit_net1 = compute_plv(spk_net1_upper, spk_net1_lower, t_sim = (params['rec_stop']-params['rec_start']), bin_size=3, transient=params['rec_start'])
        print("Phase-Locking Value (PLV) in network 1:", plv_intracircuit_net1)
        
        plv_intracircuit_net2 = compute_plv(spk_net2_upper, spk_net2_lower, t_sim = (params['rec_stop']-params['rec_start']), bin_size=3, transient=params['rec_start'])
        print("Phase-Locking Value (PLV) in network 2:", plv_intracircuit_net1)
        
        plv_intercircuit = compute_plv(spk_net1_upper, spk_net2_upper, t_sim = (params['rec_stop']-params['rec_start']), bin_size=3, transient=params['rec_start'])
        print("Phase-Locking Value (PLV) between circuits:", plv_intercircuit)
        
    else:
        spk_upper = np.sort(np.concatenate(net1_spike_data[0]))
        spk_lower = np.sort(np.concatenate(net1_spike_data[2]))
        
        plv_intracircuit_net1 = compute_plv(spk_upper, spk_lower, t_sim = (params['rec_stop']-params['rec_start']), bin_size=3, transient=params['rec_start'])
        print("Phase-Locking Value (PLV) between circuits:", plv_intracircuit_net1)
    
    
    results['PLV_intracircuit_net1'] = plv_intracircuit_net1
    
    results['net1_ISI_mean']              = np.array(net1_ISI_mean)
    results['net1_firing_rate']           = 1/(np.array(net1_ISI_mean)/1000)
    results['net1_ISI_std']               = np.array(net1_ISI_std)
    results['net1_CV']                    = np.array(net1_ISI_CV)
    
    if second_net:
        results['PLV_intercircuit']      = plv_intercircuit
        results['PLV_intracircuit_net2'] = plv_intracircuit_net2
        
        results['net2_ISI_mean']              = np.array(net2_ISI_mean)
        results['net2_firing_rate']           = 1/(np.array(net2_ISI_mean)/1000)
        results['net2_ISI_std']               = np.array(net2_ISI_std)
        results['net2_CV']                    = np.array(net2_ISI_CV)
    
    ## Write to disk
    with open("sim_results", 'wb') as f:
        pickle.dump(results, f)
    