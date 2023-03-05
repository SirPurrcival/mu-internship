import mpi4py
mpi4py.rc.thread_level = "multiple"
from bayes_opt import BayesianOptimization, UtilityFunction
from run import run_network, setup
from mpi4py import MPI
from mpi4py.futures import MPIPoolExecutor
import numpy as np

def simple_objective_function(x):
    return -x['x']**2

# Define the objective function
# def objective_function(args):#ext_rate, ext_nodes, ext_weights):
#     # Set the network parameters based on the input values
    
#     print(f"Rank {MPI.COMM_WORLD.Get_rank()} running objective_function with args: {args}")
    
#     params = setup()
#     ## Set the amount of analysis done during runtime. Minimize stuff done
#     ## for faster results
#     params['calc_lfp'] = False
#     params['verbose']  = True
#     params['opt_run']  = True
    
#     ## Change values and run the function with different parameters
#     params['ext_rate'] = args['ext_rate']
#     #params['ext_nodes'] = np.array([kwargs[x] for x in temp if 'node' in x])
#     params['ext_weights'] = np.array([args[x] for x in args if 'weight' in x])

#     # Run the spiking neuron model in NEST and compute the output measures
#     (irregularity, synchrony, firing_rate) = run_network(params)
#     ## Define target values
#     target_irregularity = 0.8
#     target_synchrony    = 0.1
#     target_firing_rate  = 5
    
#     ## Compute scores for how close to our target values we got this run
#     scores = [(target_irregularity - irregularity[i])**2 + (synchrony[i] - target_synchrony)**2 + (firing_rate[i] - target_firing_rate)**2 for i in list(range(len(irregularity)))]
        
#     # Return the negative of the score since the Bayesian optimization maximizes the objective function
#     return {'params': args, 'result': np.mean(scores)}

def objective_function(args):
    # Set the network parameters based on the input values
    
    print(f"args: {args}")
    
    params = setup()
    ## Set the amount of analysis done during runtime. Minimize stuff done
    ## for faster results
    params['calc_lfp'] = False
    params['verbose']  = False
    params['opt_run']  = True
    
    ## Change values and run the function with different parameters
    params['ext_rate'] = args['ext_rate']
    params['ext_weights'] = np.array([args[x] for x in args if 'weight' in x])

    # Run the spiking neuron model in NEST and compute the output measures
    (irregularity, synchrony, firing_rate) = run_network(params)
    ## Define target values
    target_irregularity = 0.8
    target_synchrony    = 0.1
    target_firing_rate  = 5
    
    ## Compute scores for how close to our target values we got this run
    scores = [(target_irregularity - irregularity[i])**2 + (synchrony[i] - target_synchrony)**2 + (firing_rate[i] - target_firing_rate)**2 for i in list(range(len(irregularity)))]
        
    # Return the negative of the score since the Bayesian optimization maximizes the objective function
    return {'params': args, 'result': np.mean(scores)}

# def optimize_network(optimizer, n_workers):
#     utility = UtilityFunction(kind = "ei", kappa = 1.96, xi = 0.01)
#     for i in range(20):  # perform 5 rounds of optimization
#         # Generate new parameters for each worker
#         params_list = [optimizer.suggest(utility) for _ in range(n_workers)]
    
#         # Evaluate the objective function for each parameter set in parallel
#         futures = [executor.submit(objective_function, params) for params in params_list]
        
#         results = [future.result() for future in futures]
        
#         print(f"results: {results}")
        
#         # Update the optimizer with the new results
#         for result in results:
#             optimizer.register(
#                 params=result['params'], target=result['result'])

#     # Get the best parameters and the corresponding target value
#     return params, results
        

def optimize_network(optimizer, n_workers, rank):
    utility = UtilityFunction(kind = "ei", kappa = 1.96, xi = 0.01)
    for i in range(20):  # perform 5 rounds of optimization
        # Generate new parameters for each worker
        if rank == 0:
            params_list = [optimizer.suggest(utility) for _ in range(n_workers)]
        else:
            params_list = None


        
        # Scatter the parameter sets to each worker
        print(f"This is rank {rank}")
        params_list = MPI.COMM_WORLD.scatter(params_list, root=0)
        print(params_list)
        # Evaluate the objective function for each parameter set
        # results = []
        # for params in params_list:
        #     result = objective_function(params)
        #     results.append(result)
            
        results = objective_function(params_list)

        # Gather the results from each worker
        results = MPI.COMM_WORLD.gather(results, root=0)
        results = [item for sublist in results for item in sublist]

        # Update the optimizer with the new results
        for result in results:
            optimizer.register(
                params=result['params'], target=result['result'])

    # Get the best parameters and the corresponding target value
    return optimizer.max['params'], optimizer.max['target']


#MPI.Init()
if __name__ == '__main__':
    ## Get the rank of the MPI process
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_workers = comm.Get_size()
    
    print(f"Number of workers: {n_workers}")    
    
    #pbounds = {'x': (-10000, 10000) }
    # Define the parameter space    
    pbounds = dict()
    pbounds['ext_rate'] = (0.5, 8)
    for i in range(17):
        #pbounds[f'pop{i}_stim_nodes'] = (500,2000)
        pbounds[f'pop{i}_weights'] = (1e-24, 7e0)
    
    ## Define the Bayesian optimizer
    optimizer = BayesianOptimization(
        f=None,
        pbounds=pbounds,
        #random_state=1,
    )    
    #with MPIPoolExecutor(max_workers=n_workers) as executor:
    params, target= optimize_network(optimizer, n_workers, rank)
    print(optimizer.max['params'])
    print(optimizer.max['target'])