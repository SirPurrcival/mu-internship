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

def analysis(network):
    ## read in results
    mm_file_names = [f for f in os.listdir() if os.path.isfile(f) and f.startswith("mmdata_")]
    spike_file_names = [f for f in os.listdir() if os.path.isfile(f) and f.startswith("spikes_")]
    
    spikes_gathered = []
    mmdata_gathered = []
    for i in range(len(mm_file_names)):
        with open(f"mmdata_{i}", 'rb') as f:
            mmdata_gathered.append(pickle.load(f))
        with open(f"spikes_{i}", 'rb') as f:
            spikes_gathered.append(pickle.load(f))
    
    ## join the results
    spikes = join_results(spike_gathered)
    
    ## Prepare data for graphing
    spikes = prep_spikes(spikes, network)
    
    with open("params", 'rb') as f:
       params = pickle.load( f)
    
    if params['verbose']:
        print(f"Done. Time required for simulation and gathering: {time.time() - st}")
    
    if not params['opt_run']:
        
        print("Graphing spikes...")
        ## Define colors used in the raster plot per neuron population based on label
        label = network.get_labels()
        colors = ["b" if l == "E" else "r" if l == "Pv" else "green" if l == "Sst" else "purple" for l in label]
        
        ## Plot spike data
        raster(spikes, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), label, suffix=f"{str(int(th_input)):0>4}")
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
            
            ## Only care about synaptic currents if we do LFPs
            mmdata = join_results(mm_res)
            
            print(network.multimeters)
            
            plot_LFPs(network, params, H_YX, num_neurons)                
            
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