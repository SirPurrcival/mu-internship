import random
from deap import base, creator, tools
import time
from setup import setup
import numpy as np
import pickle
import os
import subprocess

# Define the number of objectives and the number of parameters
num_objectives = 3
num_params = 18
pop_size = 50
ngen = 100

# Define the lower and upper bounds for each parameter
param_min = [0] * num_params
param_max = [1] * num_params

def simulate(args):
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
    target_firing_rate  = 5
    
    ## Compute scores for how close to our target values we got this run
    scores = [(target_irregularity - irregularity[i])**2 + (synchrony[i] - target_synchrony)**2 + (firing_rate[i] - target_firing_rate)**2 for i in list(range(len(irregularity)))]
    
    print(f"Firing rates:\n {firing_rate}")
    print(f"Duration of simulation: {time.time() - st}")
    # Return the negative of the score since the Bayesian optimization maximizes the objective function
    return -np.mean(scores)


# Define the evaluation function
def evaluate(params):
    results = simulate(params)
    # Reshape the results into a 1D array
    results_flat = results.flatten()
    # Separate the results into groups of 3 (one group per population)
    results_grouped = [results_flat[i:i+3] for i in range(0, len(results_flat), 3)]
    # Compute the average value of each group
    avg_values = [sum(group)/3 for group in results_grouped]
    return tuple(avg_values)

# Define the fitness and individual classes
creator.create("FitnessMulti", base.Fitness, weights=(-1.0,)*num_objectives)
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Define the toolbox
toolbox = base.Toolbox()

# Register the parameter initialization function
toolbox.register("param_init", random.uniform, param_min, param_max)

# Register the individual initialization function
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.param_init, n=num_params)

# Register the population initialization function
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Register the evaluation function
toolbox.register("evaluate", evaluate)

# Register the selection operator
toolbox.register("select", tools.selNSGA2)

# Register the variation operators (crossover and mutation)
toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20.0, low=param_min, up=param_max)
toolbox.register("mutate", tools.mutPolynomialBounded, eta=20.0, low=param_min, up=param_max, indpb=1.0/num_params)

# Define the main evolution loop
def main():
    # Initialize the population
    pop = toolbox.population(n=pop_size)

    # Evaluate the initial population
    fitnesses = map(toolbox.evaluate, pop)
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit

    # Begin the evolution
    for gen in range(ngen):
        # Select the next generation
        offspring = toolbox.select(pop, len(pop))

        # Apply crossover and mutation to create the offspring
        offspring = algorithms.varAnd(offspring, toolbox, cxpb=0.7, mutpb=0.3)

        # Evaluate the offspring
        fitnesses = map(toolbox.evaluate, offspring)
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit

        # Merge the parent and offspring populations
        pop = tools.selNSGA2(pop + offspring, len(pop))

        # Print some statistics
        print("Generation ", gen)
        print("  Best fitness: ", tools.selBest(pop, 1)[0].fitness.values)

    return pop

if __name__ == "__main__":
    main()