import torch
import numpy as np
#from sbi.inference.base import infer
from sbi.inference import SNPE, prepare_for_sbi, simulate_for_sbi
from sbi.utils.get_nn_models import posterior_nn
from sbi import utils as utils
from sbi import analysis as analysis
import matplotlib.pyplot as plt
import pickle
import subprocess
import os
from run import run_network
from setup import setup

# This script performs multi-round simulation-based inference for the exploration and visualisation of the parameter space 
# associated with a spiking neuron model. 
_ = torch.manual_seed(0)

num_Rounds = 1                           #number of rounds of inferences. More rounds focus the posterior around the desired observation
num_Sim = 50                            #number of simulations run per round of inference. Should be >= 50.
num_Samples = 50                         #number of smaples drawn from posterior.
#parameter space to sample
# 1st dim = external rate
# 17 dims = external weights
prior_min = [0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
prior_max = [8, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0, 7e0]
prior = utils.BoxUniform(low=torch.as_tensor(prior_min), high=torch.as_tensor(prior_max))


def simulator(args):
    #Simulator for SBI. 
    # Input: 'args' will be sampled from prior.
    # Output: array containing Irregularity, Synchrony and FR scores (results-targets) with 3*17=51 elements.
    parallel_sim = False                    #run simulation paralellized if True. I did not manage to make this work on my laptop, probably because of hardware limitations
    n_workers = 4                                      
    


    #print(args)
    params = setup()
    ## Set the amount of analysis done during runtime. Minimize stuff done
    ## for faster results
    params['calc_lfp'] = False
    params['verbose']  = False
    params['opt_run']  = True
    
    ## Change values and run the function with different parameters
    args = np.asarray(args)
    args = args.flatten()
    args = args.tolist()
    params['ext_rate'] = args[0]
    params['ext_weights'] = args[1:18:1]
    #params['K_scale'] = args['K_scale']

    ## Write parameters to file so the network can read it in
    with open("params", 'wb') as f:
        pickle.dump(params, f)
    
    ## Make sure that working directory and environment are the same
    cwd = os.getcwd()
    env = os.environ.copy()
    #
    

    #
    if parallel_sim:
    ## Run the simulation in parallel by calling the simulation script with MPI  
        command = f"mpirun -n {n_workers} --verbose python3 run.py"
    
        process = subprocess.Popen(command.split(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, 
                              cwd=cwd, env=env)
        return_code = process.wait()
    
        output, error = process.communicate()
        #print("Script exited with return code:", return_code)
        #print("Standard output:\n", output.decode())
        #print("Error output:\n", error.decode())
        process.stdout.close()
        process.stderr.close()
        #Read the results from the file
        with open("sim_results", 'rb') as f:
            data = pickle.load(f)
        irregularity, synchrony, firing_rate = data
    else:
     ## run simulation without parallelization
        irregularity, synchrony, firing_rate = run_network()
    

    ## Define target values
    target_irregularity = 0.8
    target_synchrony    = 0.1
    target_firing_rate  = 5.0
    
    ## Compute scores for how close to our target values we got this run
    scores = []
    for i in list(range(len(irregularity))):
        scores.extend([(target_irregularity - irregularity[i] ), (synchrony[i] - target_synchrony), (firing_rate[i] - target_firing_rate)])
        
    
    print(f"Scores:\n {scores}")
    # Return the negative of the score since the Bayesian optimization maximizes the objective function
    return np.asarray(scores)



def multi_round_infer(simulator, prior, x_o, num_Rounds):
# run inference over num_rounds 
    # multi-round inference to focus posterior around observation x_o
    
    #prepare simulator and prior
    simulator, prior = prepare_for_sbi(simulator, prior)
    #instantiate inference object
    inference = SNPE(prior=prior)

    posteriors = []
    #set first proposal to prior
    proposal = prior
    for _ in range(num_Rounds):
        theta, x = simulate_for_sbi(simulator, proposal, num_simulations=num_Sim)
        density_estimator = inference.append_simulations(theta, x, proposal=proposal).train()
        posterior = inference.build_posterior(density_estimator, sample_with='mcmc')
        posteriors.append(posterior)
        proposal = posterior.set_default_x(x_o)
    return posterior


#set target observation 
x_o = torch.zeros(
    51,
) 

posterior = multi_round_infer(simulator, prior, x_o, num_Rounds)

## visualise parameter space around observation
with open("logprob", 'wb') as f:
    pickle.dump(posterior,f)
samples = posterior.sample((num_Samples,), x=x_o)
log_probability = posterior.log_prob(samples, x=x_o)
fig, ax = analysis.pairplot(samples)
plt.show()