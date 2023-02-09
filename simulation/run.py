## install iCSD if needed:
## pip install git+https://github.com/espenhgn/iCSD.git

## Import libraries
import numpy as np
import matplotlib.pyplot as plt
import nest
from functions import Network, raster, rate, approximate_lfp_timecourse, get_isi, get_firing_rate, get_irregularity, get_synchrony
#import icsd

## Set nest variables
nest.ResetKernel()
#nest.SetKernelStatus({'local_num_threads': 4})  # Adapt if necessary
nest.local_num_threads = 4
nest.print_time = False
resolution = 0.1
nest.resolution = resolution

#######################################
## Set parameters for the simulation ##
#######################################

print("Begin setup")

CELLS = np.load('cells.npy', allow_pickle=True).item()

params = {
    'N':    [800, 200],                 # number of neurons in network
    'rho':  0.2,                        # fraction of inhibitory neurons
    'eps':  0.1,                        # probability to establish a connections
    'g':    4,                          # excitation-inhibition balance
    'eta':  2,                          # relative external rate
    'J':    0.1,                        # postsynaptic amplitude in mV
    'n_rec_ex':  800,                   # excitatory neurons to be recorded from
    'n_rec_in':  200,                   # inhibitory neurons to be recorded from
    'rec_start': 500.,                  # start point for data recording
    'rec_stop':  900.,                  # end points for data recording
    'sim_time': 1000.
    }


################################
## Specify synapse properties ##
################################

# delay = 1.5
# # excitatory input to receptor_type 1
# nest.CopyModel("static_synapse", "E",
#                {"weight": params['J'], "delay": delay, "receptor_type": 1})
# # inhbitiory input to receptor_type 2 (makes weight automatically postive if negative weight is supplied)
# nest.CopyModel("static_synapse", "I",
#                {"weight": params['J']*params['g'], "delay": delay, "receptor_type": 2})




########################
## Create the network ##
########################

network = Network(resolution, params['rec_start'], params['rec_stop'])

# Count of overview
num_layers = 5
num_types  = 4
num_layertypes = 17
num_parameters = 14

Nscale = 0.05
Kscale = .13
Sscale = 1
Rscale = Nscale * 0.5

ext_rate = 900*8 * Kscale

# Overview
layers = ['L1', 'L23', 'L4', 'L5', 'L6']
types  = ['E',  'Pvalb', 'Htr3a', 'Sst']
num_neurons = [776, 47386, 3876, 2807, 6683, 70387, 9502, 5455, 2640, 20740, 2186, 1958, 410, 19839, 1869, 1869, 325 ]
layertypes = ['L1_Htr3a', 'L23_E', 'L23_Pvalb', 'L23_Sst', 'L23_Htr3a' , 'L4_E', 'L4_Pvalb', 'L4_Sst', 'L4_Htr3a',  'L5_E', 'L5_Pvalb', 'L5_Sst', 'L5_Htr3a', 'L6_E', 'L6_Pvalb',  'L6_Sst', 'L6_Htr3a']
label = ['Htr','E','Pv','Sst','Htr','E','Pv','Sst','Htr','E','Pv','Sst','Htr','E','Pv','Sst','Htr']
parameters = ['adapting_threshold', 'after_spike_currents', 'asc_amps', 'asc_decay', 'asc_init', 'C_m', 'E_L', 'g', 'spike_dependent_threshold', 't_ref', 'tau_syn', 'V_m', 'V_reset', 'V_th']

# Populations
print("Populating network...")
for i in range(len(layertypes)):
    network.addpop('glif_psc', int(num_neurons[i]*Nscale), CELLS[layertypes[i]], label=label[i], nrec=int(Rscale* num_neurons[i]))


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

SS = np.array([[1.73, 0.53, 0.48, 0.57, 0.78, 0.42, 0   , 0   , 0   , 0.42, 0   , 0   , 0   , 0.42, 0   , 0   , 0   ],
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
SS *= Sscale

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

##L1 | L23e, i | L4e,i | L5e,i | L6e,i
#ext_rates = np.array([1500, 1600, 1500, 1500, 1500, 2100, 1900, 1900, 1900, 2000, 1900, 1900, 1900, 2900, 2100, 2100, 2100]) * 8 * Kscale
ext_rates = np.array([1900, 2600, 1500, 1500, 1500, 2100, 1900, 1900, 1900, 2000, 1900, 1900, 1900, 2900, 2100, 2100, 2100]) * 8 * Kscale
stim_weights = [5, 3.97, 2.2, 4.2, 2.1, 3e0, 7.5e0, 3.6, 0.8, 4.1, 4e0, 1.8, 0.02, 7.5, 2e-20, 3.2, 2.2 ]
# relative_weight = [1,                                                                                       ## Layer 1
#                     1, 3876/(3876 + 2807 + 6683), 2807/(3876 + 2807 + 6683), 6683/(3876 + 2807 + 6683),      ## Layer 23
#                     1, 9502/(9502+5455+2640), 5455/(9502+5455+2640), 2640/(9502+5455+2640),                  ## Layer 4
#                     1, 2186/(2186+1958+410), 1958/(2186+1958+410), 410/(2186+1958+410),                      ## Layer 5
#                     1, 1869/(1869+1869+325), 1869/(1869+1869+325), 325/(1869+1869+325)
#                     ]
# ext_rates = np.array([1500, 1600, 1500, 1500, 1500, 2100, 1900, 1900, 1900, 2000, 1900, 1900, 1900, 2900, 2100, 2100, 2100]) * relative_weight * 8 * Kscale

# # add stimulation
for i in range(len(ext_rates)):
    network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rates[i]}, target=i, weight=stim_weights[i])

## Connect all populations to each other according to the
## connectivity matrix and synaptic specifications
print("Connecting network...")
network.connect_all(C, S, SS)
print("Done! Starting simulation...")

## simulate
network.simulate(params['sim_time'])
print("Done! Fetching data...")

## Extract data from the network
mmdata, spikes = network.get_data()
print("Done! Graphing spikes...")

## Define colors used in the raster plot per neuron population based on label
label = network.get_labels()
colors = ["b" if l == "E" else "r" if l == "Pv" else "green" if l == "Sst" else "purple" for l in label]

## Plot spike data
raster(spikes, params['rec_start'], params['rec_stop'], colors, network.get_nrec(), label)
plt.show()

## Display the average firing rate in Hz
rate(spikes, params['rec_start'], params['rec_stop'])
print("Done! Estimating LFPs per layer...")

#################################
## LFP Approximation procedure ##
#################################

times = np.unique(mmdata[0]["times"])

## Approximate the lfp timecourse per layer
#lfp_tc_l1, all_tc = approximate_lfp_timecourse(mmdata)
lfp_tc_l1 = approximate_lfp_timecourse([mmdata[0]], times, label[0])
print("Layer 1 finished")
lfp_tc_l2 = approximate_lfp_timecourse(mmdata[1:5], times, label[1:5])
print("Layer 2/3 finished")
lfp_tc_l3 = approximate_lfp_timecourse(mmdata[5:9], times, label[5:9])
print("Layer 4 finished")
lfp_tc_l4 = approximate_lfp_timecourse(mmdata[9:13], times, label[9:13])
print("Layer 5 finished")
lfp_tc_l5 = approximate_lfp_timecourse(mmdata[13:17], times, label[13:17])
print("Layer 6 finished, plotting...")

## Correct for data loss during lfp approximation 
## (6ms due to methodological reasons, see approximation function)
t = np.argwhere(times - min(times) >= 6)
t = t.reshape(t.shape[0],)

## plot the timecourse in the recorded time window
fig, ax = plt.subplots()
ax.plot(t, lfp_tc_l1, label = "Layer 1")
ax.plot(t, lfp_tc_l2, label = "Layer 2/3")
ax.plot(t, lfp_tc_l3, label = "Layer 4")
ax.plot(t, lfp_tc_l4, label = "Layer 5")
ax.plot(t, lfp_tc_l5, label = "Layer 6")
legend = ax.legend(loc='right', bbox_to_anchor=(1.3, 0.7), shadow=False, ncol=1)
plt.show()
print("All done!")

temp = np.vstack([lfp_tc_l1, lfp_tc_l2, lfp_tc_l3, lfp_tc_l4, lfp_tc_l5])

plt.figure()
plt.imshow(temp, aspect="auto")
plt.show()

newlst = np.array([lfp_tc_l1, lfp_tc_l2, lfp_tc_l3, lfp_tc_l4, lfp_tc_l5])

print(f"Standard deviations:\n{np.std(lfp_tc_l1)}\n{np.std(lfp_tc_l2)}\n{np.std(lfp_tc_l3)}\n{np.std(lfp_tc_l4)}\n{np.std(lfp_tc_l5)}")

#icsd.CSD(lfp_tc)
