#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:18:08 2023

@author: meowlin
"""

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
    'ei'         : [],
    'k'          : [],
    'CV'         : [],
    'stdev'      : [],
    'mean'       : [],
    'firing_rate': [],
    'plv'        : []
    }

## Change values and run the function with different parameters
iteration = 1

ei_range = np.arange(1,4.1,0.25)
k_range = np.arange(0.0, 0.5, 0.05)


total_iter = len(ei_range) * len(k_range)

for ei in ei_range:
    for k in k_range:
        i = 0
        print(f"================= Iteration {iteration} of {total_iter} =================")
        print("Starting step...")
        st = time.time()
        params['E/I ratio']                = ei
        params['interlaminar_connections'] = k
        
        ## Save parameter values used
        opt_results['ei'].append(ei)
        opt_results['k'].append(k)
        
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
        
        process = subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, ##stdout=subprocess.PIPE, stderr=subprocess.PIPE, ##stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                                   cwd=cwd, env=env)
        
        return_code = process.wait()
        print("Script exited with return code:", return_code)
        
        ## Uncomment if not using DEVNULL
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
        opt_results['plv'].append(data['PLV'])
        
        print(f"Duration of iteration iteration: {time.time() - st}")
        iteration += 1
            
## Do analysis and figure stuff here

def plot_heatmap(tau_m, tau_syn_ex, data, title, round_to=0):
    ## Reshape data
    plot_pop = np.reshape(data, (len(tau_m), len(tau_syn_ex))).T
    
    ## Set values for invalid trials to black
    masked_data = np.ma.masked_where(plot_pop==-1, plot_pop)

    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(color='black')
    
    ## Plot
    fig, ax = plt.subplots()
    heatmap = ax.imshow(masked_data, cmap=cmap)
    ax.set_xticks(np.arange(len(tau_m)), tau_m)
    ax.set_yticks(np.arange(len(tau_syn_ex)), np.round(tau_syn_ex, 1))
    
    ## minor ticks
    ax.set_xticks(np.arange(len(tau_m))-0.5, minor=True)
    ax.set_yticks(np.arange(len(tau_syn_ex))-0.5, minor=True)
    
    ## Add gridlines
    ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
    
    ## Add values in white to the plot
    for i in range(len(tau_m)):
        for j in range(len(tau_syn_ex)):
            ax.text(i, j, round(plot_pop[j, i], round_to) if round_to >= 1 else int(plot_pop[j, i]),
                           ha="center", va="center", color="black", fontsize=8)
    
    ## General settings
    ax.set_title(title, fontsize=22)
    
    # Remove minor ticks
    ax.tick_params(which='minor', bottom=False, left=False)
    
    plt.xlabel(r"E/I Ratio", fontsize=18)
    plt.ylabel("Interlaminar connections", fontsize=18)
    
    plt.colorbar(heatmap)
    
    fig.tight_layout()
    plt.savefig(f"simresults/{title}.png")
    plt.show()
    

## Axis data
ei_data      = np.unique(opt_results['ei'])
k_data = np.unique(opt_results['k'])

## Get data for population 1
plv_data = opt_results['plv']
## Plot heatmaps
plot_heatmap(ei_data, k_data, plv_data, "PLV", round_to = 1)




