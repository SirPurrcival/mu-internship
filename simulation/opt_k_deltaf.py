
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
    'delta_f'    : [],
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

## Based on parameter exploration: (tau_syn_ex, tau_m)
delta_f_range = [(0.8, 18),(0.9, 16),(0.9, 17), (0.9, 18), (0.9, 19), (1.0, 16), (1.0, 18), (1.0, 19)]

k_range = np.arange(0.0, 0.25, 0.025)


total_iter = len(delta_f_range) * len(k_range)

for deep_input in deep_input_range:
    for delta_f in delta_f_range:
        for k in k_range:
            i = 0
            print(f"================= Iteration {iteration} of {total_iter} =================")
            print("Starting step...")
            st = time.time()
            params['cell_params'][2]['tau_syn_ex']  = delta_f[0]
            params['cell_params'][2]['tau_m']       = delta_f[1]
            params['interlaminar_connections']      = k
            
            params['second_net'] = False
            
            ## Save parameter values used
            opt_results['delta_f'].append(delta_f)
            opt_results['k'].append(k)
            
            ei = params['E/I ratio']
            
            params['interlaminar'] = np.array(
                ##            Target
                ##    E1              I1                E2           I2 
                    [[0.           , 0.          , k*(1-(1/ei)), k*(1-(1/ei))], ## E1
                     [0.           , 0.          , k*(1/ei)    , k*(1/ei)    ], ## I1   Source
                     [k*(1-(1/ei)) , k*(1-(1/ei)), 0.          , 0.          ], ## E2
                     [k*(1/ei)     , k*(1/ei)    , 0.          , 0.          ]] ## I2
                    )
            
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
            opt_results['plv_intra1'].append(data['PLV_intracircuit_net1'])
            
            print(f"k: {k}, ei: {ei}\nPLV: {data['PLV']}")
            
            print(f"Duration of iteration {iteration}: {time.time() - st}")
            iteration += 1
                
    ## Do analysis and figure stuff here
    
    def plot_heatmap(tau_m, tau_syn_ex, data, title, round_to=0):
        ## Reshape data
        plot_pop = np.reshape(data, (len(tau_m), len(tau_syn_ex))).T
        
        ## Set values for invalid trials to black
        masked_data = np.ma.masked_where(plot_pop==-1, plot_pop)
    
        cmap = plt.get_cmap('jet').copy()
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
        ax.invert_yaxis()
        
        
        plt.xlabel(r"$\Delta$f", fontsize=18)
        plt.ylabel("conn strength", fontsize=18)
        
        plt.colorbar(heatmap)
        
        fig.tight_layout()
        plt.savefig(f"simresults/{title}.png")
        plt.show()
        
    
    ## Axis data
    #ei_data  = np.unique(opt_results['ei'])
    delta_f_data = range(len(delta_f_range))
    k_data = np.unique(opt_results['k'])
    
    
    ## Get data for population 1
    plv_data = opt_results['plv']
    fr1_data   = [x[0] if type(x) == np.ndarray else x for x in opt_results['firing_rate']]
    fr2_data   = [x[2] if type(x) == np.ndarray else x for x in opt_results['firing_rate']]
    
    ## Plot heatmaps
    plot_heatmap(delta_f_data, k_data, plv_data, "PLV - single circuit", round_to = 1)
    plot_heatmap(delta_f_data, k_data, fr1_data, "Upper layer FR", round_to = 0)
    plot_heatmap(delta_f_data, k_data, fr2_data, "Lower layer FR", round_to = 0)
    
    


