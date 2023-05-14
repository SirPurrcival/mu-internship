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
    
    if params['verbose']:
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
    
    ## Start parallelization here?
    ## Get the rank of the MPI process
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    
    network = Network(params) #params['resolution'], params['rec_start'], params['rec_stop'], params['g'])
    
    # Populations
    if params['verbose']:
        print(f"Time required for setup: {time.time() - st}")
        print("Populating network...")
    for i in range(len(params['layer_type'])):
        network.addpop('iaf_psc_exp', params['layer_type'][i] , int(num_neurons[i]), params['cell_params'], label=params['label'][i], nrec=int(params['R_scale']* num_neurons[i]))
    
    # # add stimulation
    if params['verbose']:
        print(f"Time required to populate network: {time.time() - st}")
        print("Adding stimulation...")
    for i in range(len(ext_rates)):
        network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rates[i]}, 
                                target=params['layer_type'][i], 
                                c_specs={'rule': 'all_to_all'},
                                s_specs={'weight': params['ext_weights'][i],
                                         'delay': 1.5})
    
    ## Add DC stimulation
    th_input = params['th_in'] * 902
    network.add_stimulation(source={'type': 'poisson_generator', 'rate': th_input}, target="L4_E", 
                            c_specs={'rule': 'fixed_indegree', 'indegree': int(0.0983 * params['num_neurons'][2])},
                            s_specs={'weight': params['exc_weight'], 'delay': 1.5})
    network.add_stimulation(source={'type': 'poisson_generator', 'rate': th_input}, target="L4_E", 
                            c_specs={'rule': 'fixed_indegree', 'indegree': int(0.0619 * params['num_neurons'][3])},
                            s_specs={'weight': params['exc_weight'], 'delay': 1.5})

    network.add_stimulation(source={'type': 'poisson_generator', 'rate': th_input}, target="L4_E", 
                            c_specs={'rule': 'fixed_indegree', 'indegree': int(0.0512 * params['num_neurons'][6])},
                            s_specs={'weight': params['exc_weight'], 'delay': 1.5})
    network.add_stimulation(source={'type': 'poisson_generator', 'rate': th_input}, target="L4_E", 
                            c_specs={'rule': 'fixed_indegree', 'indegree': int(0.0196 * params['num_neurons'][7])},
                            s_specs={'weight': params['exc_weight'], 'delay': 1.5})

    
    
    ## Connect all populations to each other according to the
    ## connectivity matrix and synaptic specifications
    if params['verbose']:
        print(f"Time required for stimulation setup: {time.time() - st}")
        print("Connecting network...")
    
    network.connect_all(params['layer_type'], connections, params['syn_type'], synaptic_strength)
    
    if params['verbose']:
        print(f"Time required for connection setup: {time.time() - st}")
        
    ## Prepare LFP kernels if we care about LFPs this run
    if params['calc_lfp']:
        print("Preparing Kernels and building filters for LFP approximation...")
        H_YX = prep_LFP_kernel(params)
        network.create_fir_filters(H_YX, params)
        print(f"Time required for LFP setup {time.time() - st}")
    if params['verbose']:
        print("Done! Starting simulation...")
    
    ## simulation loop
    time_step = time.time()
    simulating = True
    simulation_time = 0
    while simulating: 
        ## End the simulation if the given time has been reached or the
        ## simulation takes too much time (excessive spiking)
        if time.time() - time_step > 200 and simulation_time > 50:
            simulating = False
            # if params['verbose']:    
            #     print("Extreme spiking, aborting run...")
            if params['verbose']:
                print(f"Simulation interrupted at {simulation_time} because it took {time.time() - time_step} seconds.")
            with open("sim_results", 'wb') as f:
                data = ([10000]*17, [10000]*17, [10000]*17)
                pickle.dump(data, f)
            #return data
        ## Exit normally
        elif simulation_time >= params['sim_time']:
            print("Simulation done!")
            simulating = False
        ## Simulate one step of 10ms
        else:
            time_step = time.time()
            network.simulate(10)
            simulation_time += 10
            print(f"Rank {rank} simulating for {simulation_time}.\nTime taken for simulating 10ms: {time.time() - time_step}s")
    
    if params['verbose']:
        print(f"Time required for simulation: {time.time() - st}")
        print("Done! Fetching data...")
    
    ## Extract data from the network
    spikes = network.get_data()
    if params['verbose']:
        print(f"Time required for fetching data: {time.time() - st}")
    
    ## end parallelization here?
    ## Gather the results from the simulations

    with open(f"tmp/spikes_{rank}", 'wb') as f:
        pickle.dump(spikes, f)
    if params['verbose']:
        print(f"rank {rank} finished!")
    
    if params['verbose']:
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
        
        ## join the results
        spikes = join_results(spikes_gathered)
        
        ## Prepare data for graphing
        spikes = prep_spikes(spikes, network)
        
        if params['verbose']:
            print(f"Done. Time required for simulation and gathering: {time.time() - st}")
        
        if not params['opt_run']:
            
            print("Graphing spikes...")
            ## Define colors used in the raster plot per neuron population based on label
            label = network.get_labels()
            ##colors = ["b" if l == "E" else "r" if l == "Pv" else "green" if l == "Sst" else "purple" for l in label]
            colors = ["b" if l == "E" else "r" for l in label]
            
            ## Plot spike data
            raster(spikes, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), label, suffix=f"{str(int(params['th_in'])):0>4}")
            plt.show()
            
            ## Display the average firing rate in Hz
            rate(spikes, params['rec_start'], params['rec_stop'])
            
            if params['verbose']:
                print(f"Time required for graphing grid: {time.time() - st}")
                
            
            #################################
            ## LFP Approximation procedure ##
            #################################
            if params['calc_lfp']:
                
                if params['verbose']:
                    print("Done! Estimating LFPs per layer...")
                
                print(network.multimeters)
                
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
