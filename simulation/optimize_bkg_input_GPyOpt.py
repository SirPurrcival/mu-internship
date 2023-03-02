from bayes_opt import BayesianOptimization
from run import run_network, setup
from mpi4py.futures import MPIPoolExecutor
import numpy as np
from mpi4py import MPI
import GPyOpt

# Define the objective function
def objective_function(kwargs):#ext_rate, ext_nodes, ext_weights):
    # Set the network parameters based on the input values
    params = setup()
    print("Running simulation")
    ## Set the amount of analysis done during runtime. Minimize stuff done
    ## for faster results
    params['calc_lfp'] = False
    params['verbose']  = False
    params['opt_run']  = True
    
    ## Change values and run the function with different parameters
    params['ext_rate'] = kwargs[0][0]
    #params['ext_nodes'] = np.array([kwargs[x] for x in kwargs if 'node' in x])
    #params['ext_weights'] = np.array([kwargs[x] for x in kwargs if 'weight' in x])

    ## Run the spiking neuron model in NEST and compute the output measures
    (irregularity, synchrony, firing_rate) = run_network(params)
    
    ## Define target values
    target_irregularity = 0.8
    target_synchrony    = 0.1
    target_firing_rate  = 5
    
    ## Compute scores for how close to our target values we got this run
    scores = [(target_irregularity - irregularity[i])**2 + (synchrony[i] - target_synchrony)**2 + (firing_rate[i] - target_firing_rate)**2 for i in list(range(len(irregularity)))]
    
    # Return the negative of the score since the Bayesian optimization maximizes the objective function
    return np.mean(scores)

def optimize_network():
    # Define the parameter space
    
    
    
    pbounds = dict()
    pbounds = [{'name': 'ext_rate', 'type': 'continuous', 'domain': (0.5, 8)}]
    # for i in range(17):
    #     #pbounds[f'pop{i}_stim_nodes'] = (500,2000)
    #     pbounds[f'pop{i}_weights'] = (1e-24, 7e0)

    # Define the number of parallel workers to use
    n_workers = 4
    
    # Define the initial set of points to evaluate
    initial_points = 4
    
    # Define the optimization algorithm and run the optimization
    optimizer = GPyOpt.methods.BayesianOptimization(f=objective_function, domain=pbounds, initial_design_numdata=initial_points,
                                                    acquisition_type='EI', verbosity=True, num_cores=4)
    optimizer.run_optimization(max_iter=100, verbosity=True)
    optimizer.plot_convergence(filename='convergence_plot.png')
    
    
    return optimizer
    # params = setup()    
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