## install iCSD if needed:
## pip install git+https://github.com/espenhgn/iCSD.git

## Import libraries
import numpy as np
import matplotlib.pyplot as plt
import nest
from functions import Network, raster, rate, approximate_lfp_timecourse
#import icsd

## Set nest variables
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads': 4})  # Adapt if necessary
nest.print_time = False
resolution = 0.1
nest.resolution = resolution

#######################################
## Set parameters for the simulation ##
#######################################

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
    'rec_start': 600.,                  # start point for data recording
    'rec_stop':  800.,                  # end points for data recording
    'sim_time': 1000.
    }

# neuron_params={
#     "V_m": -79.0417277018229,
#     "V_th": -49.63934810542196,
#     "g": 3.4780284104908676,
#     "E_L": -79.0417277018229,
#     "C_m": 60.72689987399939,
#     "t_ref": 1.4500000000000002,
#     "V_reset": -79.0417277018229,
#     "asc_init": [
#         0.0,
#         0.0
#     ],
#     "asc_decay": [
#         0.029999999999999992,
#         0.3
#     ],
#     "asc_amps": [
#         -23.825265478178427,
#         -292.06473034028727
#     ],
#     "tau_syn": [
#         5.5,
#         8.5,
#         2.8,
#         5.8
#     ],
#     "spike_dependent_threshold": False,
#     "after_spike_currents": True,
#     "adapting_threshold": False
# }

# np2 = {
#     "V_m": -78.0417277018229,
#     "V_th": -49.63934810542196,
#     "g": 3.4780284104908676,
#     "E_L": -79.0417277018229,
#     "C_m": 60.72689987399939,
#     "t_ref": 1.4500000000000002,
#     "V_reset": -79.0417277018229,
#     "asc_init": [
#         0.0,
#         0.0
#     ],
#     "asc_decay": [
#         0.029999999999999992,
#         0.3
#     ],
#     "asc_amps": [
#         -23.825265478178427,
#         -292.06473034028727
#     ],
#     "tau_syn": [
#         5.5,
#         8.5,
#         2.8,
#         5.8
#     ],
#     "spike_dependent_threshold": False,
#     "after_spike_currents": True,
#     "adapting_threshold": False
# }

# np3={
#     "V_m": -79.0417277018229,
#     "V_th": -49.63934810542196,
#     "g": 4.4780284104908676,
#     "E_L": -79.0417277018229,
#     "C_m": 60.72689987399939,
#     "t_ref": 1.4500000000000002,
#     "V_reset": -79.0417277018229,
#     "asc_init": [
#         0.0,
#         0.0
#     ],
#     "asc_decay": [
#         0.029999999999999992,
#         0.3
#     ],
#     "asc_amps": [
#         -23.825265478178427,
#         -292.06473034028727
#     ],
#     "tau_syn": [
#         5.5,
#         8.5,
#         2.8,
#         5.8
#     ],
#     "spike_dependent_threshold": False,
#     "after_spike_currents": True,
#     "adapting_threshold": False
# }
################################
## Specify synapse properties ##
################################

delay = 1.5
# excitatory input to receptor_type 1
nest.CopyModel("static_synapse", "E",
               {"weight": params['J'], "delay": delay, "receptor_type": 1})
# inhbitiory input to receptor_type 2 (makes weight automatically postive if negative weight is supplied)
nest.CopyModel("static_synapse", "I",
               {"weight": params['J']*-params['g'], "delay": delay, "receptor_type": 2})


ext_rate = 900*8

########################
## Create the network ##
########################

network = Network(resolution, params['rec_start'], params['rec_stop'])

## Distribute the point neurons in space to prepare for LFP approximation
pos_ex = nest.spatial.free(nest.random.normal(mean=0.5, std=1.),
                        num_dimensions=3)
pos_in = nest.spatial.free(nest.random.normal(mean=0., std=1.),
                        num_dimensions=3)

# Count of overview
num_layers = 5
num_types  = 4
num_layertypes = 17
num_parameters = 14

# Overview
layers = ['L1', 'L23', 'L4', 'L5', 'L6']
types  = ['E',  'Pvalb', 'Htr3a', 'Sst']
layertypes = ['L1_Htr3a', 'L23_E', 'L23_Pvalb', 'L23_Htr3a', 'L23_Sst', 'L4_E', 'L4_Pvalb', 'L4_Htr3a', 'L4_Sst', 'L5_E', 'L5_Pvalb', 'L5_Htr3a', 'L5_Sst', 'L6_E', 'L6_Pvalb', 'L6_Htr3a', 'L6_Sst']
parameters = ['adapting_threshold', 'after_spike_currents', 'asc_amps', 'asc_decay', 'asc_init', 'C_m', 'E_L', 'g', 'spike_dependent_threshold', 't_ref', 'tau_syn', 'V_m', 'V_reset', 'V_th']

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

# Connectivity matrix layertype X layertype
C = np.array([[0.656, 0.356, 0.093, 0.068, 0.4644, 0.148, 0, 0, 0, 0.148, 0, 0, 0, 0.148, 0, 0, 0],
              [0, 0.16, 0.395, 0.182, 0.105, 0.016, 0.083, 0.083, 0.083, 0.083, 0.081, 0.102, 0, 0, 0, 0, 0],
              [0.024, 0.411, 0.451, 0.03, 0.22, 0.05, 0.05, 0.05, 0.05, 0.07, 0.073, 0, 0, 0, 0, 0, 0],
              [0.279, 0.424, 0.857, 0.082, 0.77, 0.05, 0.05, 0.05, 0.05, 0.021, 0, 0, 0, 0, 0, 0, 0],
              [0, 0.087, 0.02, 0.625, 0.028, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0.14, 0.100, 0.1, 0.1, 0.243, 0.43, 0.571, 0.571, 0.104, 0.101, 0.128, 0.05, 0.032, 0, 0, 0],
              [0, 0.25, 0.050, 0.05, 0.05, 0.437, 0.451, 0.03, 0.22, 0.088, 0.091, 0.03, 0.03, 0, 0, 0, 0],
              [0.241, 0.25, 0.050, 0.05, 0.05, 0.351, 0.857, 0.082, 0.77, 0.026, 0.03, 0, 0.03, 0, 0, 0, 0],
              [0, 0.25, 0.050, 0.05, 0.05, 0.351, 0.02, 0.625, 0.028, 0, 0.03, 0.03, 0.03, 0, 0, 0, 0],
              [0.017, 0.021, 0.05, 0.05, 0.05, 0.007, 0.05, 0.05, 0.05, 0.116, 0.083, 0.063, 0.105, 0.047, 0.03, 0.03, 0.03],
              [0, 0, 0.102, 0, 0, 0, 0.034, 0.03, 0.03, 0.455, 0.361, 0.03, 0.22, 0.03, 0.01, 0.01, 0.01],
              [0.203, 0.169, 0, 0.017, 0, 0.056, 0.03, 0.006, 0.03, 0.317, 0.857, 0.04, 0.77, 0.03, 0.01, 0.01, 0.01],
              [0, 0, 0, 0, 0, 0.03, 0.03, 0.03, 0.03, 0.125, 0.02, 0.625, 0.02, 0.03, 0.01, 0.01, 0.01],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.012, 0.01, 0.01, 0.01, 0.026, 0.145, 0.1, 0.1],
              [0, 0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1, 0.03, 0.03, 0.03, 0.1, 0.08, 0.1, 0.08],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03, 0.03, 0.03, 0.03, 0.1, 0.05, 0.05, 0.05],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0.03, 0.03, 0.03, 0.03, 0.1, 0.05, 0.05, 0.03]])

# Populations
network.addpop('glif_psc', 776, CELLS['L1_Htr3a'], pos_ex, label="I", nrec=776)

network.addpop('glif_psc', 47386, CELLS['L23_E'], pos_ex, label="E", nrec=47386)
network.addpop('glif_psc', 3876, CELLS['L23_Pvalb'], pos_ex, label="I", nrec=3876)
network.addpop('glif_psc', 2807, CELLS['L23_Sst'], pos_ex, label="I", nrec=2807)
network.addpop('glif_psc', 6683, CELLS['L23_Htr3a'], pos_ex, label="I", nrec=6683)

network.addpop('glif_psc', 70387, CELLS['L4_E'], pos_ex, label="E", nrec=70387)
network.addpop('glif_psc', 9502, CELLS['L4_Pvalb'], pos_ex, label="I", nrec=9502)
network.addpop('glif_psc', 5455, CELLS['L4_Sst'], pos_ex, label="I", nrec=5455)
network.addpop('glif_psc', 2640, CELLS['L4_Htr3a'], pos_ex, label="I", nrec=2640)

network.addpop('glif_psc', 20740, CELLS['L5_E'], pos_ex, label="E", nrec=20740)
network.addpop('glif_psc', 2186, CELLS['L5_Pvalb'], pos_ex, label="I", nrec=2186)
network.addpop('glif_psc', 1958, CELLS['L5_Sst'], pos_ex, label="I", nrec=1958)
network.addpop('glif_psc', 410, CELLS['L5_Htr3a'], pos_ex, label="I", nrec=410)

network.addpop('glif_psc', 19839, CELLS['L6_E'], pos_ex, label="E", nrec=19839)
network.addpop('glif_psc', 1869, CELLS['L6_Pvalb'], pos_ex, label="I", nrec=1869)
network.addpop('glif_psc', 1869, CELLS['L6_Sst'], pos_ex, label="I", nrec=1869)
network.addpop('glif_psc', 325, CELLS['L6_Htr3a'], pos_ex, label="I", nrec=325)

# # add stimulation
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=0) # to excitatory population
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=1) # to inhibitory population
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=2)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=3)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=4) # to excitatory population
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=5) # to inhibitory population
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=6)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=7)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=8)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=9)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=10)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=11)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=12)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=13)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=14)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=15)
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=16)

# ## Add populations to the network
# network.addpop('glif_psc', params['N'][0], [neuron_params, np2, np3], pos_ex, label="E", nrec=800)
# network.addpop('glif_psc', params['N'][1], [neuron_params, np2, np3], pos_in, label="I", nrec=200)

# network.addpop('glif_psc', params['N'][0], [neuron_params, np2, np3], pos_ex, label="E", nrec=800)
# network.addpop('glif_psc', params['N'][1], [neuron_params, np2, np3], pos_in, label="I", nrec=200)

# network.addpop('glif_psc', params['N'][0], [neuron_params, np2, np3], pos_ex, label="E", nrec=800)
# network.addpop('glif_psc', params['N'][1], [neuron_params, np2, np3], pos_in, label="I", nrec=200)

# network.addpop('glif_psc', 400, [neuron_params, np2, np3], pos_ex, label="E", nrec=400)
# network.addpop('glif_psc', 100, [neuron_params, np2, np3], pos_in, label="I", nrec=100)

# network.addpop('glif_psc', params['N'][0], [neuron_params, np2, np3], pos_ex, label="E", nrec=800)
# network.addpop('glif_psc', params['N'][1], [neuron_params, np2, np3], pos_in, label="I", nrec=200)

# ## Define connectivity matrix
# conn_matrix = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
#                         [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
# syn_matrix = np.array([["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
#                        ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
#                        ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
#                        ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
#                        ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
#                        ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
#                        ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
#                        ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
#                        ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
#                        ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"]])

## Connect all populations to each other according to the
## connectivity matrix and synaptic specifications
network.connect_all(C, S)

## simulate
print("Starting simulation...")
network.simulate(params['sim_time'])
print("Done! Fetching data...")

## Extract data from the network
mmdata, spikes = network.get_data()
print("Done! Graphing spikes...")

## Define colors used in the raster plot per neuron population based on label
label = network.get_labels()
colors = ["b" if l == "E" else "r" for l in label]

## Plot spike data
raster(spikes, params['rec_start'], params['rec_stop'], colors)
plt.show()

## Display the average firing rate in Hz
rate(spikes, params['rec_start'], params['rec_stop'])
print("Done! Estimating LFPs per layer...")

times = np.unique(mmdata[0]["times"])

## Approximate the lfp timecourse per layer
#lfp_tc_l1, all_tc = approximate_lfp_timecourse(mmdata)
lfp_tc_l1 = approximate_lfp_timecourse(mmdata[0:2], times, label[0:2])
print("Layer 1 finished")
lfp_tc_l2 = approximate_lfp_timecourse(mmdata[2:4], times, label[2:4])
print("Layer 2 finished")
lfp_tc_l3 = approximate_lfp_timecourse(mmdata[4:6], times, label[4:6])
print("Layer 3 finished")
lfp_tc_l4 = approximate_lfp_timecourse(mmdata[6:8], times, label[6:8])
print("Layer 4 finished")
lfp_tc_l5 = approximate_lfp_timecourse(mmdata[8:10], times, label[8:10])
print("Layer 5 finished, plotting...")

## Correct for data loss during lfp approximation 
## (6ms due to methodological reasons, see approximation function)
t = np.argwhere(times - min(times) >= 6)
t = t.reshape(t.shape[0],)

times = mmdata[0]["times"][t]

## plot the timecourse in the recorded time window
plt.plot(t, lfp_tc_l1)
plt.plot(t, lfp_tc_l2)
plt.plot(t, lfp_tc_l3)
plt.plot(t, lfp_tc_l4)
plt.plot(t, lfp_tc_l5)
plt.show
print("All done!")

newlst = np.array([lfp_tc_l1, lfp_tc_l2, lfp_tc_l3, lfp_tc_l4, lfp_tc_l5])

#icsd.CSD(lfp_tc)
