#!/usr/bin/python3
print("Starting sim script")
######################
## Import libraries ##
######################
import mpi4py as mpi
mpi.rc.thread_level = "multiple"

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
import nest
import scipy as sp
from functions import Network, plot_LFPs, raster, rate, approximate_lfp_timecourse, get_irregularity, get_synchrony, get_firing_rate, join_results, prep_spikes
#import icsd
import time
from prep_LFP_kernel import prep_LFP_kernel
import os
import pickle

def run_network():
    ## Load config created by setup()
    with open("params", 'rb') as f:
       params = pickle.load( f)
    
    ## Start parallelization here?
    ## Get the rank of the MPI process
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    
    if params['verbose'] and rank == 0:
        print("Begin setup")
        ## get the start time
        st = time.time()
    
    ########################
    ## Set NEST Variables ##
    ########################
    nest.ResetKernel()
    nest.local_num_threads = 32 ## adapt if necessary
    nest.print_time = False
    nest.resolution = params['resolution']
    nest.set_verbosity("M_WARNING")
    nest.overwrite_files = True
    ## Path relative to working directory
    #nest.data_path = ""
    ## Some random prefix that is given to all files (i.e. the trial number)
    #nest.data_prefix = ""
    
    #####################
    # Scale the network #
    #####################
    
    synaptic_strength = params['syn_strength'] * params['syn_scale']
    connections       = params['connectivity'] * params['K_scale']
    num_neurons       = params['num_neurons']  * params['N_scale']
    ext_rates         = params['ext_nodes'] * params['ext_rate']
    
    ########################
    ## Create the network ##
    ########################
    

    
    network = Network(params) #params['resolution'], params['rec_start'], params['rec_stop'], params['g'])
    
    # Populations
    if params['verbose'] and rank == 0:
        print(f"Time required for setup: {time.time() - st}")
        print("Populating network...")
    for i in range(len(params['layer_type'])):
        network.addpop('iaf_psc_exp', params['layer_type'][i] , int(num_neurons[i]), params['cell_params'], label=params['label'][i], record_from_pop=True, nrec=int(params['R_scale']* num_neurons[i]))
    
    # # add stimulation
    if params['verbose'] and rank == 0:
        print(f"Time required to populate network: {time.time() - st}")
        print("Adding stimulation...")
    for i in range(len(ext_rates)):
        network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rates[i], 'start': 0, 'stop': params['sim_time']}, 
                                target=params['layer_type'][i], 
                                c_specs={'rule': 'all_to_all'},
                                s_specs={'weight': params['ext_weights'][i],
                                         'delay': 1.5})


    
    
    ## Connect all populations to each other according to the
    ## connectivity matrix and synaptic specifications
    if params['verbose'] and rank == 0:
        print(f"Time required for stimulation setup: {time.time() - st}")
        print("Connecting network...")
    
    network.connect_all(params['layer_type'], connections, params['syn_type'], synaptic_strength)
    
    ## Add thalamic input
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
        print(f"Time required for connection setup: {time.time() - st}")
        
    ## Prepare LFP kernels if we care about LFPs this run
    if params['calc_lfp']:
        print("Preparing Kernels and building filters for LFP approximation...")
        H_YX = prep_LFP_kernel(params)
        network.create_fir_filters(H_YX, params)
        print(f"Time required for LFP setup {time.time() - st}")
    if params['verbose'] and rank == 0:
        print("Done! Starting simulation...")
    
    ## simulation loop
    time_step = time.time()
    simulating = True
    simulation_time = 0
    while simulating: 
        ## Exit normally
        if simulation_time >= params['sim_time']:
            print("Simulation done!")
            simulating = False
        ## Simulate one step of 10ms
        else:
            time_step = time.time()
            network.simulate(50)
            simulation_time += 50
            if rank == 0:
                print(f"Simulating for {simulation_time}.\nTime taken for simulating 10ms: {time.time() - time_step}s")
    
    if params['verbose'] and rank == 0:
        print(f"Time required for simulation: {time.time() - st}")
        print("Done! Fetching data...")
    
    ## Extract data from the network
    mmdata, spikes = network.get_data()
    if params['verbose'] and rank == 0:
        print(f"Time required for fetching data: {time.time() - st}")
    
    ## end parallelization here?
    ## Gather the results from the simulations

    with open(f"tmp/spikes_{rank}", 'wb') as f:
        pickle.dump(spikes, f)
    with open(f"tmp/mmdata_{rank}", 'wb') as f:
        pickle.dump(mmdata, f)
        
    if params['verbose'] and rank == 0:
        print(f"rank {rank} finished!")
    
    if params['verbose'] and rank == 0:
        print("Done, gathering results and preparing data...")
    
    comm.barrier()
    
    ## Gather results and write to disk
    if rank == 0:
        ## read in results
        spike_file_names = [f for f in os.listdir("tmp") if f.startswith("spikes_")]
        
        spikes_gathered = []
        for i in range(len(spike_file_names)):
            with open(f"tmp/spikes_{i}", 'rb') as f:
                spikes_gathered.append(pickle.load(f))
                
        ## read in results
        mm_file_names = [f for f in os.listdir("tmp") if f.startswith("mmdata_")]
        
        mm_gathered = []
        for i in range(len(mm_file_names)):
            with open(f"tmp/mmdata_{i}", 'rb') as f:
                mm_gathered.append(pickle.load(f))
        
        ## join the results
        spikes = join_results(spikes_gathered)
        v_membrane = join_results(mm_gathered)
        
        ## Prepare data for graphing
        spikes = prep_spikes(spikes, network)
        
        
        vm_data = []
        for i in v_membrane:
            ## For each population
            times = np.array(i['times'])
            senders = i['senders']
            v = np.array(i['V_m'])
            
            # Bin the membrane potentials based on time steps
            bin_edges = np.arange(times.min(), times.max() + 0.126, 0.125)
            bin_counts, _ = np.histogram(times, bins=bin_edges)
            bin_means, _ = np.histogram(times, bins=bin_edges, weights=v)
            bin_means[bin_counts == 0] = np.nan  # Set empty bins to NaN to avoid division by zero
            
            # Compute the mean for each bin
            mean_v = bin_means / bin_counts

            vm_data.append(mean_v)
            # # for j in np.unique(times):
            # #     np.where(v == j)
        vm_data = np.array(vm_data)
        
        
            

        if params['verbose'] and rank == 0:
            print(f"Done. Time required for simulation and gathering: {time.time() - st}")
            print(np.shape(vm_data))
        
        if not params['opt_run']:
            
            print("Graphing spikes...")
            ## Define colors used in the raster plot per neuron population based on label
            label = network.get_labels()
            ##colors = ["b" if l == "E" else "r" if l == "Pv" else "green" if l == "Sst" else "purple" for l in label]
            colors = ["tab:blue" if l == "E" else "crimson" for l in label]
            
            ## Plot spike data
            raster(spikes, vm_data, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), label, suffix=f"{str(int(params['th_in'])):0>4}")
            plt.show()
            ## Display the average firing rate in Hz
            rate(spikes, params['rec_start'], params['rec_stop'])
            
            if params['verbose'] and rank == 0:
                print(f"Time required for graphing grid: {time.time() - st}")
                
            # ## Scipy spectrogram
            # def create_spectrogram(data, fs):
                
            #     _, _,  _, spectrogram = plt.specgram(data, Fs=fs)
            #     plt.colorbar(label='Power Spectral Density (dB/Hz)')
            #     plt.xlabel('Time (s)')
            #     plt.ylabel('Frequency (Hz)')
            #     plt.savefig("simresults/spectrogram.png")
                
            #     return spectrogram
            
            # # Example usage
            # data = vm_data
            # fs = nest.resolution
            
            # spectrogram = create_spectrogram(data, fs)

            # a, b, c = sp.signal.spectrogram(vm_data, fs=nest.resolution)
            
            # plt.plot(c)
            # plt.savefig("simresults/spectrogram.png")            
            #################################
            ## LFP Approximation procedure ##
            #################################
            if params['calc_lfp']:
                
                if params['verbose'] and rank == 0:
                    print("Done! Estimating LFPs per layer...")
                
                
                plot_LFPs(network, params, H_YX, num_neurons)                
                
                
        irregularity = [get_irregularity(population) for population in spikes]
        firing_rate  = [get_firing_rate(population, params['rec_start'], params['rec_stop']) for population in spikes]
        synchrony    = [get_synchrony(population, params['rec_start'], params['rec_stop']) for population in spikes]
    
        ## Write results to file
        with open("sim_results", 'wb') as f:
            data = (irregularity, synchrony, firing_rate)
            pickle.dump(data, f)
        return (irregularity, synchrony, firing_rate)
     
#from setup import setup
#setup()
run_network()
