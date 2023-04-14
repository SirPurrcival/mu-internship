from bayes_opt import BayesianOptimization, UtilityFunction
#from run import run_network
import numpy as np
import subprocess
import pickle
from setup import setup
import os
import time


def objective_function(**args):
    # Set the network parameters based on the input values
    n_workers = 16
    
    print("Starting step...")
    st = time.time()
    
    params = setup()
    ## Set the amount of analysis done during runtime. Minimize stuff done
    ## for faster results
    params['calc_lfp'] = False
    params['verbose']  = False
    params['opt_run']  = True
    
    ## Change values and run the function with different parameters
    params['ext_rate'] = args['ext_rate']
    params['ext_weights'] = np.array([args[x] for x in args if 'weight' in x])

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
    # output, error = process.communicate()
    # print("Standard output:\n", output.decode())
    # print("Error output:\n", error.decode())
    
    #process.stdout.close()
    #process.stderr.close()
    
    # Read the results from the file
    with open("sim_results", 'rb') as f:
        data = pickle.load(f)
    irregularity, synchrony, firing_rate = data
    
    if sum(irregularity) >= 169.999 and sum(synchrony) >= 169.999 and sum(firing_rate) >= 169.999:
        print("Run was aborted due to excessive spiking")
    
    ## Define target values
    target_irregularity = 0.9
    target_synchrony    = 0.05
    target_firing_rate  = 2
    
    ## Compute scores for how close to our target values we got this run
    scores = [(target_irregularity - irregularity[i])**2 + (synchrony[i] - target_synchrony)**2 + (firing_rate[i] - target_firing_rate)**2 for i in list(range(len(irregularity)))]
    
    print(f"Firing rates:\n {firing_rate}")
    print(f"Duration of simulation: {time.time() - st}")
    # Return the negative of the score since the Bayesian optimization maximizes the objective function
    return -np.mean(scores)

def optimize_network(optimizer):
    uf = UtilityFunction(kind = "ucb", kappa = 4, xi = 0.2) # xi = 0.01 , kappa = 1.96
    
    optimizer.set_gp_params(alpha=1e-5, n_restarts_optimizer=5, normalize_y=False)
    
    optimizer.maximize(
            init_points=40,
            n_iter=100,
            acquisition_function=uf
        )
    # best = None
    # for i in range(200):  # perform 5 rounds of optimization
    #     # Generate new parameters for each worker
        
        
    #     params = optimizer.suggest(uf)
        
    #     result = objective_function(params)
        
    #     optimizer.register(
    #         params=result['params'], target=result['result'])
        
        
    #     if best == None:
    #         best = result['result']
    #     elif best < result['result']:
    #         best = result['result']
    #     # Get the best parameters and the corresponding target value
    #     print(f"Result of run {i}: {result['result']}\nCurrent best: {best}")
        
    ##Return the best parameters
    return optimizer.max['params']
        
if __name__ == '__main__':
    # Define the parameter space    
    pbounds = dict()
    pbounds['ext_rate'] = (3.0, 8.0)
    for i in range(17):
        #pbounds[f'pop{i}_stim_nodes'] = (500,2000)
        pbounds[f'pop{i}_weights'] = (1e-24, 14e0)
    
    # ## Define the Bayesian optimizer
    # optimizer = BayesianOptimization(
    #     f=None,
    #     pbounds=pbounds,
    #     #random_state=1,
    # )
    ##Define the Bayesian optimizer
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=1,
    )
    
    #with MPIPoolExecutor(max_workers=n_workers) as executor:

    try:
        params= optimize_network(optimizer)
    except:
        print("An error occurred (probably duplicate value). Writing current best to file and exiting...")
        print(optimizer.max['params'])
        with open("simresults/best_params", 'wb') as f:
            pickle.dump(optimizer.max['params'], f)
        
    print(optimizer.max['params'])
    with open("simresults/best_params", 'wb') as f:
        pickle.dump(optimizer.max['params'], f)
    