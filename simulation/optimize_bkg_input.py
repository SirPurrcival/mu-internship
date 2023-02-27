from bayes_opt import BayesianOptimization
from run import run_network, setup
import numpy as np
import matplotlib.pyplot as plt


# Define the objective function
def objective_function(**kwargs):#ext_rate, ext_nodes, ext_weights):
    # Set the network parameters based on the input values
    params = setup()
    
    temp = kwargs
    ## Change values and run the function with different parameters
    params['ext_rate'] = temp['ext_rate']
    params['ext_nodes'] = np.array([kwargs[x] for x in temp if 'node' in x])
    params['ext_weights'] = np.array([kwargs[x] for x in temp if 'weights' in x])

    # Run the spiking neuron model in NEST and compute the output measures
    (irregularity, synchrony, firing_rate) = run_network(params)
    
    ## Define target values
    target_irregularity = 0.8
    target_synchrony    = 0.1
    min_firing_rate     = 2
    max_firing_rate     = 10
    
    ## Compute scores for how close to our target values 
    scores = []
    for i in range(len(irregularity)):
        score = (target_irregularity - irregularity[i])**2 + (synchrony[i] - target_synchrony)**2 + max(0, (firing_rate[i] - max_firing_rate)**2) + max(0, (min_firing_rate - firing_rate[i])**2)
        scores.append(score)
    
    # Return the negative of the score since the Bayesian optimization maximizes the objective function
    return -np.mean(score)

def optimize_network():
    # Define the parameter space
    
    pbounds = dict()
    pbounds['ext_rate'] = (0.5, 8)
    for i in range(17):
        pbounds[f'pop{i}_stim_nodes'] = (500,2000)
        pbounds[f'pop{i}_weights'] = (1e-24, 7e0)

    # Define the Bayesian optimizer
    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        random_state=1,
    )

    # Choose an acquisition function
    optimizer.maximize(
        init_points=5,
        n_iter=20,
        acq='ei',
    )
    optimizer.maximize()
    
    # Visualize the progress
    optimizer.plot_progress()
    plt.show()
    
    # Get the optimal set of parameters
    optimal_params = optimizer.max['params']

    params = setup()
    
    return optimal_params
    params['ext_rate'] = optimal_params
    # # Verify that the output measures meet the desired targets
    # network_parameters = {
    #     'parameter1': optimal_params['parameter1'],
    #     'parameter2': optimal_params['parameter2'],
    #     # Add additional parameters here
    # }
    (irregularity, synchrony, firing_rate) = objective_function(params)
    
    target_irregularity = 0.8
    target_synchrony    = 0.1
    min_firing_rate     = 2
    max_firing_rate     = 10
    
    for i in len(irregularity):
        if not (irregularity[i] >= target_irregularity and synchrony[i] <= target_synchrony and firing_rate[i] >= min_firing_rate and firing_rate[i] <= max_firing_rate):
            print("Targets not met, returning last set of parameters")
            return optimal_params
        else:
            print("Optimization was successful, returning optimal parameters")
            return optimal_params
        
nyan = optimize_network()