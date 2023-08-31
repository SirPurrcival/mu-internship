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
params['second_net'] = False
params['deep_in'] = 0

opt_results = {
    'inh_nodes'       : [],
    'net1_CV'         : [],
    'net1_stdev'      : [],
    'net1_mean'       : [],
    'net1_firing_rate': [],
    'plv_intra1'      : []
    }

## Change values and run the function with different parameters
iteration = 1

node_range = range(100, 29, -1)


total_iter = len(node_range)

for inh_nodes in node_range:
    ## Which population do we want to change?
    i = 0
    print(f"================= Iteration {iteration} of {total_iter} =================")
    print("Starting step...")
    st = time.time()
    
    params['ext_rate']  = 30
    params['ext_nodes'] = np.array([100, inh_nodes, 100, inh_nodes])
    
    ## Save parameter values used
    opt_results['inh_nodes'].append(inh_nodes)

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

    opt_results['net1_CV'].append(data['net1_CV'])
    opt_results['net1_mean'].append(data['net1_ISI_mean'])
    opt_results['net1_stdev'].append(data['net1_ISI_std'])
    opt_results['net1_firing_rate'].append(data['net1_firing_rate'])

    opt_results['plv_intra1'].append(data['PLV_intracircuit_net1'])

    print(f"Duration of iteration {iteration}: {time.time() - st}")
    iteration += 1

## Do analysis and figure stuff here
## Axis data

node_data = np.flip(np.unique(opt_results['inh_nodes']))

## Get data for population 1
fr_data   = [x[0] if type(x) == np.ndarray else x for x in opt_results['net1_firing_rate']]
CV_data   = [x[0] if type(x) == np.ndarray else x for x in opt_results['net1_CV']]
## Plot heatmaps
fig, ax = plt.subplots()
ax.plot(node_data, fr_data)
ax.set_title(r"Firing rate", fontsize=20)
plt.xlabel(r"Inhibitory nodes", fontsize=18)
plt.ylabel("Pop firing rate", fontsize=18)
plt.show()

fig, ax = plt.subplots()
ax.plot(node_data, CV_data)
ax.set_title(r"$C_v$", fontsize=20)
plt.xlabel(r"Inhibitory nodes", fontsize=18)
plt.ylabel("$C_v$", fontsize=18)
plt.show()