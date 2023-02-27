#!/usr/bin/python3

######################
## Import libraries ##
######################
import numpy as np
import matplotlib.pyplot as plt
import nest
from functions import Network, raster, rate, approximate_lfp_timecourse, get_isi, get_irregularity, get_synchrony, get_firing_rate
#import icsd
import time

## Set the scaling factors for the model
Nscale = .05                  ## Scaling the number of neurons in
Kscale = .15                  ## Scaling the number of connections 
Sscale = 1.                  ## Scaling the synaptic strength
Rscale = Nscale * 0.1        ## Scaling the number of neurons we record from

## get the start time
st = time.time()

########################
## Set NEST Variables ##
########################

nest.ResetKernel()
nest.local_num_threads = 256 ## adapt if necessary
nest.print_time = False
resolution = 0.1
nest.resolution = resolution

nest.overwrite_files = True
## Path relative to working directory
#nest.data_path = ""
## Some random prefix that is given to all files (i.e. the trial number)
#nest.data_prefix = ""

#######################################
## Set parameters for the simulation ##
#######################################

print("Begin setup")
CELLS = np.load('cells.npy', allow_pickle=True).item()

## OVERVIEW
num_layers = 5
num_types  = 4
num_layertypes = 17
num_parameters = 14

layers = ['L1', 'L23', 'L4', 'L5', 'L6']
types  = ['E',  'Pvalb', 'Htr3a', 'Sst']
num_neurons = [776, 47386, 3876, 2807, 6683, 70387, 9502, 5455, 2640, 20740, 2186, 1958, 410, 19839, 1869, 1869, 325 ]
layertypes = ['L1_Htr3a', 'L23_E', 'L23_Pvalb', 'L23_Sst', 'L23_Htr3a' , 'L4_E', 'L4_Pvalb', 'L4_Sst', 'L4_Htr3a',  'L5_E', 'L5_Pvalb', 'L5_Sst', 'L5_Htr3a', 'L6_E', 'L6_Pvalb',  'L6_Sst', 'L6_Htr3a']
label = ['Htr','E','Pv','Sst','Htr','E','Pv','Sst','Htr','E','Pv','Sst','Htr','E','Pv','Sst','Htr']
parameters = ['adapting_threshold', 'after_spike_currents', 'asc_amps', 'asc_decay', 'asc_init', 'C_m', 'E_L', 'g', 'spike_dependent_threshold', 't_ref', 'tau_syn', 'V_m', 'V_reset', 'V_th']



ext_rate = 900*8 * Kscale   ## Noise rate (Nr. of noise inputs * Frequency * Scaling factor)

## Recording and simulation parameters
params = {
    'rec_start': 600.,                  # start point for data recording
    'rec_stop':  800.,                  # end points for data recording
    'sim_time': 1000.                   # Time the network is simulated in ms
    }

################################################################
## Specify connectivity in and between layers and populations ##
################################################################

# Connectivity matrix layertype X layertype
C = np.array([[0.656, 0.356, 0.093, 0.068, 0.4644, 0.148, 0    , 0    , 0    , 0.148, 0    , 0    , 0    , 0.148, 0    , 0    , 0    ],
              [0    , 0.16 , 0.395, 0.182, 0.105 , 0.016, 0.083, 0.083, 0.083, 0.083, 0.081, 0.102, 0    , 0    , 0    , 0    , 0    ],
              [0.024, 0.411, 0.451, 0.03 , 0.22  , 0.05 , 0.05 , 0.05 , 0.05 , 0.07 , 0.073, 0    , 0    , 0    , 0    , 0    , 0    ],
              [0.279, 0.424, 0.857, 0.082, 0.77  , 0.05 , 0.05 , 0.05 , 0.05 , 0.021, 0    , 0    , 0    , 0    , 0    , 0    , 0    ],
              [0    , 0.087, 0.02 , 0.625, 0.028 , 0.05 , 0.05 , 0.05 , 0.05 , 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0    ],
              [0    , 0.14 , 0.100, 0.1  , 0.1   , 0.243, 0.43 , 0.571, 0.571, 0.104, 0.101, 0.128, 0.05 , 0.032, 0    , 0    , 0    ],
              [0    , 0.25 , 0.050, 0.05 , 0.05  , 0.437, 0.451, 0.03 , 0.22 , 0.088, 0.091, 0.03 , 0.03 , 0    , 0    , 0    , 0    ],
              [0.241, 0.25 , 0.050, 0.05 , 0.05  , 0.351, 0.857, 0.082, 0.77 , 0.026, 0.03 , 0    , 0.03 , 0    , 0    , 0    , 0    ],
              [0    , 0.25 , 0.050, 0.05 , 0.05  , 0.351, 0.02 , 0.625, 0.028, 0    , 0.03 , 0.03 , 0.03 , 0    , 0    , 0    , 0    ],
              [0.017, 0.021, 0.05 , 0.05 , 0.05  , 0.007, 0.05 , 0.05 , 0.05 , 0.116, 0.083, 0.063, 0.105, 0.047, 0.03 , 0.03 , 0.03 ],
              [0    , 0    , 0.102, 0    , 0     , 0    , 0.034, 0.03 , 0.03 , 0.455, 0.361, 0.03 , 0.22 , 0.03 , 0.01 , 0.01 , 0.01 ],
              [0.203, 0.169, 0    , 0.017, 0     , 0.056, 0.03 , 0.006, 0.03 , 0.317, 0.857, 0.04 , 0.77 , 0.03 , 0.01 , 0.01 , 0.01 ],
              [0    , 0    , 0    , 0    , 0     , 0.03 , 0.03 , 0.03 , 0.03 , 0.125, 0.02 , 0.625, 0.02 , 0.03 , 0.01 , 0.01 , 0.01 ],
              [0    , 0    , 0    , 0    , 0     , 0    , 0    , 0    , 0    , 0.012, 0.01 , 0.01 , 0.01 , 0.026, 0.145, 0.1  , 0.1  ],
              [0    , 0.1  , 0    , 0    , 0     , 0.1  , 0    , 0    , 0    , 0.1  , 0.03 , 0.03 , 0.03 , 0.1  , 0.08 , 0.1  , 0.08 ],
              [0    , 0    , 0    , 0    , 0     , 0    , 0    , 0    , 0    , 0.03 , 0.03 , 0.03 , 0.03 , 0.1  , 0.05 , 0.05 , 0.05 ],
              [0    , 0    , 0    , 0    , 0     , 0    , 0    , 0    , 0    , 0.03 , 0.03 , 0.03 , 0.03 , 0.1  , 0.05 , 0.05 , 0.03 ]])
C *= Kscale

################################
## Specify synapse properties ##
################################

S = np.array([["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E"], 
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E"], 
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E"], 
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E"], 
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
              ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"]])

synaptic_strength = np.array([[1.73, 0.53, 0.48, 0.57, 0.78, 0.42, 0   , 0   , 0   , 0.42, 0   , 0   , 0   , 0.42, 0   , 0   , 0   ],
                               [0   , 0.36, 1.49, 0.86, 1.31, 0.34, 1.39, 0.69, 0.91, 0.74, 1.32, 0.53, 0   , 0   , 0   , 0   , 0   ],
                               [0.37, 0.48, 0.68, 0.42, 0.41, 0.56, 0.68, 0.42, 0.41, 0.2 , 0.79, 0   , 0   , 0   , 0   , 0   , 0   ],
                               [0.47, 0.31, 0.5 , 0.15, 0.52, 0.3 , 0.5 , 0.15, 0.52, 0.22, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                               [0   , 0.28, 0.18, 0.32, 0.37, 0.29, 0.18, 0.32, 0.37, 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                               [0   , 0.78, 1.39, 0.69, 0.91, 0.83, 1.29, 0.51, 0.51, 0.63, 1.25, 0.52, 0.91, 0.96, 0   , 0   , 0   ],
                               [0   , 0.56, 0.68, 0.42, 0.41, 0.64, 0.68, 0.42, 0.41, 0.73, 0.94, 0.42, 0.41, 0   , 0   , 0   , 0   ],
                               [0.39, 0.3 , 0.5 , 0.15, 0.52, 0.29, 0.5 , 0.15, 0.52, 0.28, 0.45, 0.28, 0.52, 0   , 0   , 0   , 0   ],
                               [0   , 0.29, 0.18, 0.32, 0.37, 0.29, 0.18, 0.32, 0.37, 0   , 0.18, 0.33, 0.37, 0   , 0   , 0   , 0   ],
                               [0.76, 0.47, 1.25, 0.52, 0.91, 0.38, 1.25, 0.52, 0.91, 0.75, 1.2 , 0.52, 1.31, 0.4 , 2.5 , 0.52, 1.31],
                               [0   , 0   , 0.51, 0   , 0   , 0   , 0.94, 0.42, 0.41, 0.81, 1.19, 0.41, 0.41, 0.81, 1.19, 0.41, 0.41],
                               [0.31, 0.25, 0   , 0.39, 0   , 0.28, 0.45, 0.28, 0.52, 0.27, 0.4 , 0.4 , 0.52, 0.27, 0.4 , 0.4 , 0.52],
                               [0   , 0   , 0   , 0   , 0   , 0.29, 0.18, 0.33, 0.37, 0.28, 0.18, 0.33, 0.37, 0.28, 0.18, 0.33, 0.37],
                               [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.23, 2.5 , 0.52, 1.31, 0.94, 3.8 , 0.52, 1.31],
                               [0   , 0.81, 0   , 0   , 0   , 0.81, 0   , 0   , 0   , 0.81, 1.19, 0.41, 0.41, 0.81, 1.19, 0.41, 0.41],
                               [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.27, 0.4 , 0.4 , 0.52, 0.27, 0.4 , 0.4 , 0.52],
                               [0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.28, 0.18, 0.33, 0.37, 0.28, 0.18, 0.33, 0.37]])
synaptic_strength *= Sscale



########################
## Create the network ##
########################

network = Network(resolution, params['rec_start'], params['rec_stop'])

# Populations
print(f"Time required for setup: {time.time() - st}")
print("Populating network...")
for i in range(len(layertypes)):
    network.addpop('glif_psc', int(num_neurons[i]*Nscale), CELLS[layertypes[i]], label=label[i], nrec=int(Rscale* num_neurons[i]))

##L1 | L23e, i | L4e,i | L5e,i | L6e,i
#ext_rates = np.array([1500, 1600, 1500, 1500, 1500, 2100, 1900, 1900, 1900, 2000, 1900, 1900, 1900, 2900, 2100, 2100, 2100]) * 8 * Kscale ## original
ext_rates = np.array([1400, 1000, 1400, 1200, 1100, 1500, 1800, 1800, 1800, 1700, 1900, 1900, 1900, 2600, 2100, 2100, 2100])* 4 * 1 * Kscale
stim_weights = [5, 
            1.97, 2.2, 4.2, 1.5, 
            3e0, 7.5e0, 3.6, 0.8, 
            3.58, 4e0, 1.8, 0.02, 
            6.5, 2e-20, 3.2, 2.2 ]
# relative_weight = [1,                                                                                       ## Layer 1
#                     1, 3876/(3876 + 2807 + 6683), 2807/(3876 + 2807 + 6683), 6683/(3876 + 2807 + 6683),      ## Layer 23
#                     1, 9502/(9502+5455+2640), 5455/(9502+5455+2640), 2640/(9502+5455+2640),                  ## Layer 4
#                     1, 2186/(2186+1958+410), 1958/(2186+1958+410), 410/(2186+1958+410),                      ## Layer 5
#                     1, 1869/(1869+1869+325), 1869/(1869+1869+325), 325/(1869+1869+325)
#                     ]
#ext_rates = np.array([1500, 1200, 1500, 1500, 1500, 2100, 1900, 1900, 1900, 2000, 1900, 1900, 1900, 2900, 2100, 2100, 2100]) * relative_weight * 8 * Kscale

# # add stimulation
for i in range(len(ext_rates)):
    network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rates[i]}, target=i, weight=stim_weights[i])

## Connect all populations to each other according to the
## connectivity matrix and synaptic specifications
print(f"Time required to populate network: {time.time() - st}")
print("Connecting network...")
network.connect_all(C, S, synaptic_strength)
print(f"Time required for connection setup: {time.time() - st}")
print("Done! Starting simulation...")

## simulate
network.simulate(params['sim_time'])
print(f"Time required for simulation: {time.time() - st}")
print("Done! Fetching data...")

## Extract data from the network
mmdata, spikes = network.get_data()
print(f"Time required for fetching data: {time.time() - st}")
print("Done! Graphing spikes...")

## Define colors used in the raster plot per neuron population based on label
label = network.get_labels()
colors = ["b" if l == "E" else "r" if l == "Pv" else "green" if l == "Sst" else "purple" for l in label]

## Plot spike data
raster(spikes, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), label)
plt.show()

## Display the average firing rate in Hz
rate(spikes, params['rec_start'], params['rec_stop'])
print(f"Time required for graphing grid: {time.time() - st}")
print("Done! Estimating LFPs per layer...")

#################################
## LFP Approximation procedure ##
#################################

times = np.unique(mmdata[0]["times"])

## append labels to data
for i in range(len(mmdata)):
    mmdata[i].update( {"label":label[i]})

## Approximate the lfp timecourse per layer
lfp_tc_l1 = approximate_lfp_timecourse([mmdata[0]], times)
print("Layer 1 finished")
lfp_tc_l23 = approximate_lfp_timecourse(mmdata[1:5], times)
print("Layer 2/3 finished")
lfp_tc_l4 = approximate_lfp_timecourse(mmdata[5:9], times)
print("Layer 4 finished")
lfp_tc_l5 = approximate_lfp_timecourse(mmdata[9:13], times)
print("Layer 5 finished")
lfp_tc_l6 = approximate_lfp_timecourse(mmdata[13:17], times)
print("Layer 6 finished, plotting...")
print(f"Time required for layer estimation procedure: {time.time() - st}")

##################
## Timing Stuff ##
##################
# # get the start time
# st = time.time()

# ## The stuff to time
# lfp_tc_l3 = approximate_lfp_timecourse(mmdata[5:9], times)

# # get the end time
# et = time.time()

# # get the execution time
# elapsed_time = et - st
# print(elapsed_time)

## Current best: 5.36

## Correct for data loss during lfp approximation 
## (6ms due to methodological reasons, see approximation function)
t = np.argwhere(times - min(times) >= 6)
t = t.reshape(t.shape[0],)

## plot the timecourse in the recorded time window
fig, ax = plt.subplots()
ax.plot(t, lfp_tc_l1, label = "Layer 1")
ax.plot(t, lfp_tc_l23, label = "Layer 2/3")
ax.plot(t, lfp_tc_l4, label = "Layer 4")
ax.plot(t, lfp_tc_l5, label = "Layer 5")
ax.plot(t, lfp_tc_l6, label = "Layer 6")
plt.show()

legend = ax.legend(loc='right', bbox_to_anchor=(1.3, 0.7), shadow=False, ncol=1)
plt.show()
plt.savefig('simresults/LFP_approximation.png')

temp = np.vstack([lfp_tc_l1, lfp_tc_l23, lfp_tc_l4, lfp_tc_l5, lfp_tc_l6])

plt.figure()
plt.imshow(temp, aspect="auto")
plt.show()
plt.savefig('simresults/vstack.png')

print(f"Time required for plotting and final time: {time.time() - st}")
print("All done!")

newlst = np.array([lfp_tc_l1, lfp_tc_l23, lfp_tc_l4, lfp_tc_l5, lfp_tc_l6])

# results = []
# for spk in spikes:
#     print(f"Average firing rate: {get_firing_rate(spk, params['rec_start'], params['rec_stop'])} \nIrregularity: {get_irregularity(spk)} \nSynchrony: {get_synchrony(spk)}")
#     results.append([get_firing_rate(spk, params['rec_start'], params['rec_stop']), get_irregularity(spk), get_synchrony(spk)])
#     #return results
#print(get_irregularity(spikes))

#run_model()
#icsd.CSD(lfp_tc)