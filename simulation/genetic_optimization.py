import random
from functools import partial
from deap import algorithms, base, creator, tools

# Define the number of objectives, number of parameters, and parameter bounds
NUM_OBJECTIVES = 17*3
NUM_PARAMETERS = 18
PARAMETER_BOUNDS = [(0.0, 8.0)]
syn_strength_bounds = [(1e-24, 15.0)] * 17

# Define the fitness function that evaluates a candidate solution
def evaluate(params):
    results = simulate(params)
    return tuple(results),

# Define the genetic algorithm parameters
POPULATION_SIZE = 100
GENERATIONS = 50
CROSSOVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.2

# Create the fitness and individual classes
creator.create("FitnessMulti", base.Fitness, weights=(-1.0,) * NUM_OBJECTIVES)
creator.create("Individual", list, fitness=creator.FitnessMulti)

# Create the toolbox for the genetic algorithm
toolbox = base.Toolbox()
toolbox.register("attr_float", random.uniform, PARAMETER_BOUNDS[0][0], PARAMETER_BOUNDS[0][1])
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=NUM_PARAMETERS)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)

# Set up the genetic algorithm and run it
random.seed(0)
population = toolbox.population(n=POPULATION_SIZE)
hof = tools.ParetoFront()
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("avg", tools.mean)
stats.register("std", tools.std)
stats.register("min", tools.min)
stats.register("max", tools.max)

algorithms.eaSimple(population, toolbox, cxpb=CROSSOVER_PROBABILITY, mutpb=MUTATION_PROBABILITY,
                    ngen=GENERATIONS, stats=stats, halloffame=hof)

# Print the best solution(s)
best_params = hof[0]
best_results = evaluate(best_params)[0]
print("Best parameters:", best_params)
print("Best results:", best_results)
