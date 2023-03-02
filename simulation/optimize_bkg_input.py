# from bayes_opt import BayesianOptimization
from run import run_network, setup
# from mpi4py.futures import MPIPoolExecutor
import numpy as np
from skopt import gp_minimize
# from mpi4py import MPI
# from skopt.utils import use_named_args
# from skopt.space import Real

def simple_objective_function(x):
    print(x)
    return x[0]**2 + x[0] - x[0]**3 

# Define the objective function
def objective_function(args):#ext_rate, ext_nodes, ext_weights):
    # Set the network parameters based on the input values
    params = setup()
    
    ## Set the amount of analysis done during runtime. Minimize stuff done
    ## for faster results
    params['calc_lfp'] = False
    params['verbose']  = False
    params['opt_run']  = True
    
    ## Change values and run the function with different parameters
    params['ext_rate'] = args[0]
    #params['ext_nodes'] = np.array([kwargs[x] for x in temp if 'node' in x])
    params['ext_weights'] = args[1:]

    # Run the spiking neuron model in NEST and compute the output measures
    (irregularity, synchrony, firing_rate) = run_network(params)
    ## Define target values
    target_irregularity = 0.8
    target_synchrony    = 0.1
    target_firing_rate  = 5
    
    ## Compute scores for how close to our target values we got this run
    scores = [(target_irregularity - irregularity[i])**2 + (synchrony[i] - target_synchrony)**2 + (firing_rate[i] - target_firing_rate)**2 for i in list(range(len(irregularity)))]
    # scores = []
    # for i in range(len(irregularity)):
    #     score = (target_irregularity - irregularity[i])**2 + (synchrony[i] - target_synchrony)**2 + (firing_rate[i] - target_firing_rate)**2
    #     scores.append(score)
    
    # Return the negative of the score since the Bayesian optimization maximizes the objective function
    return np.mean(scores)

def optimize_network():
    # Define the parameter space    
    pbounds = dict()
    pbounds['ext_rate'] = (0.5, 8)
    for i in range(17):
        #pbounds[f'pop{i}_stim_nodes'] = (500,2000)
        pbounds[f'pop{i}_weights'] = (1e-24, 7e0)

    #pbounds = [(-500, 500)]

    # # Get the rank of the MPI process
    # comm = MPI.COMM_WORLD
    # rank = comm.Get_rank()
    # n_workers = comm.Get_size()
    
    # Define the Bayesian optimizer
    # optimizer = BayesianOptimization(
    #     f=objective_function,
    #     pbounds=pbounds,
    #     random_state=1,
    # )
    
    res = gp_minimize(objective_function,                   # the function to minimize
                  list(pbounds.values()),
                  #pbounds,
                  acq_func="EI",                            # the acquisition function
                  n_calls=300,                               # the number of evaluations of f
                  n_initial_points=10,                      # the number of random initialization points
                  random_state=1234,
                  acq_optimizer="lbfgs",
                  n_jobs=4,
                  verbose=True
                  )
    
    # n_workers = 4  # number of MPI processes
    # with MPIPoolExecutor(max_workers=n_workers) as executor:
    # # Choose an acquisition function
    #     optimizer.maximize(
    #         init_points=30,
    #         n_iter=70,
    #         acq='ei',
    #         executor=executor
    #     )
    #optimizer.maximize()
    
    # Get the optimal set of parameters
    # optimal_params = optimizer.max['params']

    # params = setup()
    
    return res
    # params['ext_rate'] = optimal_params
    # # # Verify that the output measures meet the desired targets
    # # network_parameters = {
    # #     'parameter1': optimal_params['parameter1'],
    # #     'parameter2': optimal_params['parameter2'],
    # #     # Add additional parameters here
    # # }
    # (irregularity, synchrony, firing_rate) = objective_function(params)
    
    # target_irregularity = 0.8
    # target_synchrony    = 0.1
    # min_firing_rate     = 2
    # max_firing_rate     = 10
    
    # for i in len(irregularity):
    #     if not (irregularity[i] >= target_irregularity and synchrony[i] <= target_synchrony and firing_rate[i] >= min_firing_rate and firing_rate[i] <= max_firing_rate):
    #         print("Targets not met, returning last set of parameters")
    #         return optimal_params
    #     else:
    #         print("Optimization was successful, returning optimal parameters")
    #         return optimal_params
        
nyan = optimize_network()