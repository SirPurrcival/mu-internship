#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:36:13 2023

@author: meowlin
"""
import subprocess
import os
import time
import pickle
import numpy as np

import matplotlib.pyplot as plt

from setup import setup

# Set the amount of nodes to be used
n_workers = 1

## Load parameters
params = setup()
## Set the amount of analysis done during runtime. Minimize stuff done
## for faster results
params['calc_lfp'] = False
params['verbose']  = True
params['opt_run']  = True

opt_results = {
    'tau_m'      : [],
    'tau_syn_ex' : [],
    'CV'         : [],
    'stdev'      : [],
    'mean'       : [],
    'firing_rate': []
    }

## Change values and run the function with different parameters
iteration = 1
total_iter = len(range(19,31,1)) * len(np.arange(0.7, 1.3, 0.1))

for tau_m in range(18,30,1):
    for tau_syn_ex in np.arange(0.7, 1.4, 0.1):
        i = 0
        print(f"================= Iteration {iteration} of {total_iter} =================")
        print("Starting step...")
        st = time.time()
        params['cell_params'][i]['tau_m']      = tau_m
        params['cell_params'][i]['tau_syn_ex'] = tau_syn_ex
        
        ## Save parameter values used
        opt_results['tau_m'].append(tau_m)
        opt_results['tau_syn_ex'].append(tau_syn_ex)
        
        #params['cell_params'][i]['tau_syn_in'] = tau_syn_in
        
        ## Write parameters to file so the network can read it in
        with open("params", 'wb') as f:
            pickle.dump(params, f)
        
        ## Make sure that working directory and environment are the same
        cwd = os.getcwd()
        env = os.environ.copy()
        
        ## Run the simulation in parallel by calling the simulation script with MPI
        ## If the amount of workers is equal to one, we dont need MPI
        if n_workers == 1:
            command = "python3 run.py"
        else:
            command = f"mpirun -n {n_workers} --verbose python3 run.py"
        
        process = subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, ##stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   cwd=cwd, env=env)
        
        return_code = process.wait()
        print("Script exited with return code:", return_code)
        # output, error = process.communicate()
        # print("Standard output:\n", output.decode())
        # print("Error output:\n", error.decode())
        
        # process.stdout.close()
        # process.stderr.close()
        
        # Read results from file
        with open("sim_results", 'rb') as f:
            data = pickle.load(f)
            
        opt_results['CV'].append(data['CV'])
        opt_results['mean'].append(data['ISI_mean'])
        opt_results['stdev'].append(data['ISI_std'])
        opt_results['firing_rate'].append(data['firing_rate'])
        
        print(f"Duration of iteration iteration: {time.time() - st}")
        iteration += 1
            
## Do analysis and figure stuff here

def plot_heatmap(tau_m, tau_syn_ex, data, title):
    ## Reshape data
    plot_pop = np.reshape(data, (len(tau_m), len(tau_syn_ex))).T
    
    ## Set values for invalid trials to black
    masked_data = np.ma.masked_where(plot_pop==-1, plot_pop)

    cmap = plt.get_cmap('viridis')
    cmap.set_bad(color='black')
    
    ## Plot
    fig, ax = plt.subplots()
    ax.imshow(masked_data, cmap=cmap)
    ax.set_xticks(np.arange(len(tau_m)), tau_m)
    ax.set_yticks(np.arange(len(tau_syn_ex)), np.round(tau_syn_ex, 2))
    
    ## Add values in white to the plot
    for i in range(len(tau_m)):
        for j in range(len(tau_syn_ex)):
            ax.text(i, j, round(plot_pop[j, i], 1),
                           ha="center", va="center", color="w")
    
    ## General settings
    ax.set_title(title, fontsize=22)
    plt.xlabel(r"$\tau_{m}$", fontsize=18)
    plt.ylabel(r"$\tau_{syn-ex}$", fontsize=18)
    fig.tight_layout()
    plt.savefig(f"simresults/{title}.png")
    plt.show()

## Axis data
tau_m_data      = np.unique(opt_results['tau_m'])
tau_syn_ex_data = np.unique(opt_results['tau_syn_ex'])

## Get data for population 1
CV_data   = [x[0] if type(x) == list else x for x in opt_results['CV']]
mean_data = [x[0] if type(x) == list else x for x in opt_results['mean']]
std_data  = [x[0] if type(x) == list else x for x in opt_results['stdev']]
fr_data   = [x[0] if type(x) == list else x for x in opt_results['firing_rate']]

## Plot heatmaps
plot_heatmap(tau_m_data, tau_syn_ex_data, CV_data, "Coefficient of variation")
plot_heatmap(tau_m_data, tau_syn_ex_data, mean_data, "Mean")
plot_heatmap(tau_m_data, tau_syn_ex_data, std_data, "Standard Deviation")
plot_heatmap(tau_m_data, tau_syn_ex_data, fr_data, "Firing rate")



