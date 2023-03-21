#!/usr/bin/python3

######################
## Import libraries ##
######################
import mpi4py
mpi4py.rc.thread_level = "multiple"

from mpi4py import MPI

import numpy as np
import matplotlib.pyplot as plt
import nest
from functions import Network, plot_LFPs, raster, rate, approximate_lfp_timecourse, get_irregularity, get_synchrony, get_firing_rate, join_results, prep_spikes
#import icsd
import time
from prep_LFP_kernel import prep_LFP_kernel

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
    
    network = Network(params['resolution'], params['rec_start'], params['rec_stop'])
    
    # Populations
    if params['verbose']:
        print(f"Time required for setup: {time.time() - st}")
        print("Populating network...")
    for i in range(len(params['layer_type'])):
        network.addpop('glif_psc', params['layer_type'][i] , int(num_neurons[i]), params['cell_type'][params['layer_type'][i]], label=params['label'][i], nrec=int(params['R_scale']* num_neurons[i]))
    
    ##L1 | L23e, i | L4e,i | L5e,i | L6e,i
    #ext_rates = np.array([1500, 1600, 1500, 1500, 1500, 2100, 1900, 1900, 1900, 2000, 1900, 1900, 1900, 2900, 2100, 2100, 2100]) * 8 * Kscale ## original
    
    # relative_weight = [1,                                                                                       ## Layer 1
    #                     1, 3876/(3876 + 2807 + 6683), 2807/(3876 + 2807 + 6683), 6683/(3876 + 2807 + 6683),      ## Layer 23
    #                     1, 9502/(9502+5455+2640), 5455/(9502+5455+2640), 2640/(9502+5455+2640),                  ## Layer 4
    #                     1, 2186/(2186+1958+410), 1958/(2186+1958+410), 410/(2186+1958+410),                      ## Layer 5
    #                     1, 1869/(1869+1869+325), 1869/(1869+1869+325), 325/(1869+1869+325)
    #                     ]
    #ext_rates = np.array([1500, 1200, 1500, 1500, 1500, 2100, 1900, 1900, 1900, 2000, 1900, 1900, 1900, 2900, 2100, 2100, 2100]) * relative_weight * 8 * Kscale
    
    # # add stimulation
    if params['verbose']:
        print(f"Time required to populate network: {time.time() - st}")
        print("Adding stimulation...")
    for i in range(len(ext_rates)):
        network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rates[i]}, target=params['layer_type'][i], weight=params['ext_weights'][i])
    
    ## Connect all populations to each other according to the
    ## connectivity matrix and synaptic specifications
    if params['verbose']:
        print(f"Time required for stimulation setup: {time.time() - st}")
        print("Connecting network...")
    
    network.connect_all(params['layer_type'], connections, params['syn_type'], synaptic_strength)
    
    if params['verbose']:
        print(f"Time required for connection setup: {time.time() - st}")
        
    if params['calc_lfp']:
        print("Preparing Kernels and building filters for LFP approximation...")
        H_YX = prep_LFP_kernel(params)
        network.create_fir_filters(H_YX, params)
        print(f"Time required for LFP setup {time.time() - st}")
        print("Done! Starting simulation...")
    
    ## simulate
    network.simulate(params['sim_time'])
    if params['verbose']:
        print(f"Time required for simulation: {time.time() - st}")
        print("Done! Fetching data...")
    
    ## Extract data from the network
    mmdata, spikes = network.get_data()
    if params['verbose']:
        print(f"Time required for fetching data: {time.time() - st}")
    
    ## end parallelization here?
    ## Gather the results from the simulations
    
    if params['verbose']:
        print("Done, gathering results and preparing data...")
    
    mm_res    = MPI.COMM_WORLD.gather(mmdata, root=0)
    spike_res = MPI.COMM_WORLD.gather(spikes, root=0)
    
    
    if rank == 0:
        ## join the results
        # print(f"Size of population recordings: {network.get_nrec()}")
        
        # for i in range(len(network.get_pops())):
        #     print(f"network size: {min(list(network.get_pops()[i].get(['global_id']).values())[0])}")
        #     print(f"network size: {max(list(network.get_pops()[i].get(['global_id']).values())[0])}")
        
        #IDs = list(self.__populations[i].get(['global_id']).values())[0]
        
        
        mmdata = join_results(mm_res)
        spikes = join_results(spike_res)
        
        ## Prepare data for graphing
        spikes = prep_spikes(spikes, network)
        
        if not params['opt_run']:
            print("Done! Graphing spikes...")
            ## Define colors used in the raster plot per neuron population based on label
            label = network.get_labels()
            colors = ["b" if l == "E" else "r" if l == "Pv" else "green" if l == "Sst" else "purple" for l in label]
            
            ## Plot spike data
            raster(spikes, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), label)
            plt.show()
            
            ## Display the average firing rate in Hz
            rate(spikes, params['rec_start'], params['rec_stop'])
            
            if params['verbose']:
                print(f"Time required for graphing grid: {time.time() - st}")
                print("Done! Estimating LFPs per layer...")
            
            #################################
            ## LFP Approximation procedure ##
            #################################
            if params['calc_lfp']:
                pass
                plot_LFPs(network, params, H_YX, num_neurons)
                
                print(network.multimeters)
                
                
                
                times = np.unique(mmdata[0]["times"])
                
                ## append labels to data
                for i in range(len(mmdata)):
                    mmdata[i].update( {"label":label[i]})
                
                ## Approximate the lfp timecourse per layer
                lfp_tc_l1 = approximate_lfp_timecourse([mmdata[0]], times)
                if params['verbose']:
                    print("Layer 1 finished")
                lfp_tc_l23 = approximate_lfp_timecourse(mmdata[1:5], times)
                if params['verbose']:
                    print("Layer 2/3 finished")
                lfp_tc_l4 = approximate_lfp_timecourse(mmdata[5:9], times)
                if params['verbose']:
                    print("Layer 4 finished")
                lfp_tc_l5 = approximate_lfp_timecourse(mmdata[9:13], times)
                if params['verbose']:
                    print("Layer 5 finished")
                lfp_tc_l6 = approximate_lfp_timecourse(mmdata[13:17], times)
                if params['verbose']:
                    print("Layer 6 finished, plotting...")
                    print(f"Time required for layer estimation procedure: {time.time() - st}")
                
                ## Correct for data loss during lfp approximation 
                ## (6ms due to methodological reasons, see approximation function)
                t = np.argwhere(times - min(times) >= 6)
                t = t.reshape(t.shape[0],)
                
                ## plot the timecourse in the recorded time window
                fig, ax = plt.subplots()
                ax.plot(t, lfp_tc_l1, label = "Layer 1")
                ax.plot(t, lfp_tc_l23, label = "Layer 2/3")
                ax.plot(t, lfp_tc_l4, label = "Layer 4")
                ax.plot(t, lfp_tc_l5, label = "Layer 5")
                ax.plot(t, lfp_tc_l6, label = "Layer 6")
                plt.show()
                
                legend = ax.legend(loc='right', bbox_to_anchor=(1.3, 0.7), shadow=False, ncol=1)
                plt.show()
                plt.savefig('simresults/LFP_approximation.png')
                
                temp = np.vstack([lfp_tc_l1, lfp_tc_l23, lfp_tc_l4, lfp_tc_l5, lfp_tc_l6])
                
                plt.figure()
                plt.imshow(temp, aspect="auto")
                plt.show()
                plt.savefig('simresults/vstack.png')
                
                if params['verbose']:
                    print(f"Time required for plotting and final time: {time.time() - st}")
                    print("All done!")
                
                newlst = np.array([lfp_tc_l1, lfp_tc_l23, lfp_tc_l4, lfp_tc_l5, lfp_tc_l6])
                
        irregularity = [get_irregularity(population) for population in spikes]
        firing_rate  = [get_firing_rate(population, params['rec_start'], params['rec_stop']) for population in spikes]
        synchrony    = [get_synchrony(population, params['rec_start'], params['rec_stop']) for population in spikes]
        
        ## Write results to file
        with open("sim_results", 'wb') as f:
            data = (irregularity, synchrony, firing_rate)
            pickle.dump(data, f)
        return (irregularity, synchrony, firing_rate)
     
from setup import setup
setup()
nya = run_network()
