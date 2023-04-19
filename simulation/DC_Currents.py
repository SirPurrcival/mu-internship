#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 13:50:40 2023

@author: meowlin
"""

from setup import setup
import time
import os
import pickle
import subprocess
import numpy as np


params = setup()

params['calc_lfp'] = False
params['verbose']  = False
params['opt_run']  = False

n_workers = 16

fr_list = []

for th in range(0,3,0.01):
    params['th_in'] = 900 * th
    
    print(f"Starting run with thalamic input of {th}")
    st = time.time()

    ## Write parameters to file so the network can read it in
    with open("params", 'wb') as f:
        pickle.dump(params, f)
    
    ## Make sure that working directory and environment are the same
    cwd = os.getcwd()
    env = os.environ.copy()

    ## Run the simulation in parallel by calling the simulation script with MPI  
    command = f"mpirun -n {n_workers} --verbose python3 run.py"
    
    process = subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,##stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                               cwd=cwd, env=env)
    
    return_code = process.wait()
    print("Script exited with return code:", return_code)
    
    print(f"Duration of simulation: {time.time() - st}")
    with open("sim_results", 'rb') as f:
        data = pickle.load(f)
        
    irregularity, synchrony, firing_rate = data
    fr_list.append((th ,np.mean(firing_rate)))

with open("simresults/firing_rates", 'wb') as f:
    pickle.dump(fr_list, f)

