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

# Set the amount of nodes to be used
n_workers = 1

## Load parameters
params = setup()
## Set the amount of analysis done during runtime. Minimize stuff done
## for faster results
params['calc_lfp'] = False
params['verbose']  = True
params['opt_run']  = True

results = {
    'tau_m'     : [],
    'tau_syn_ex': [],
    'results'   : []
    }

## Change values and run the function with different parameters
for tau_m in range(20,26,1):
    for tau_syn_ex in np.arange(0.7, 1.3, 0.1):
        for i in range(2):
            print("Starting step...")
            st = time.time()
            params['cell_params'][i]['tau_m']      = tau_m
            params['cell_params'][i]['tau_syn_ex'] = tau_syn_ex
            
            ## Save parameter values used
            results['tau_m'].append(tau_m)
            results['tau_syn_ex'].append(tau_syn_ex)
            
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
            
            process = subprocess.Popen(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE, ##stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                       cwd=cwd, env=env)
            
            return_code = process.wait()
            print("Script exited with return code:", return_code)
            output, error = process.communicate()
            print("Standard output:\n", output.decode())
            print("Error output:\n", error.decode())
            
            process.stdout.close()
            process.stderr.close()
            
            # Read results from file
            with open("sim_results", 'rb') as f:
                data = pickle.load(f)
                
            results['results'].append(data)

            ## Define target values
            
            print(f"Duration of simulation: {time.time() - st}")
            
## Do analysis and figure stuff here