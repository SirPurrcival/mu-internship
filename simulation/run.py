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

neuron_params={
    "V_m": -79.0417277018229,
    "V_th": -49.63934810542196,
    "g": 3.4780284104908676,
    "E_L": -79.0417277018229,
    "C_m": 60.72689987399939,
    "t_ref": 1.4500000000000002,
    "V_reset": -79.0417277018229,
    "asc_init": [
        0.0,
        0.0
    ],
    "asc_decay": [
        0.029999999999999992,
        0.3
    ],
    "asc_amps": [
        -23.825265478178427,
        -292.06473034028727
    ],
    "tau_syn": [
        5.5,
        8.5,
        2.8,
        5.8
    ],
    "spike_dependent_threshold": False,
    "after_spike_currents": True,
    "adapting_threshold": False
}

np2 = {
    "V_m": -78.0417277018229,
    "V_th": -49.63934810542196,
    "g": 3.4780284104908676,
    "E_L": -79.0417277018229,
    "C_m": 60.72689987399939,
    "t_ref": 1.4500000000000002,
    "V_reset": -79.0417277018229,
    "asc_init": [
        0.0,
        0.0
    ],
    "asc_decay": [
        0.029999999999999992,
        0.3
    ],
    "asc_amps": [
        -23.825265478178427,
        -292.06473034028727
    ],
    "tau_syn": [
        5.5,
        8.5,
        2.8,
        5.8
    ],
    "spike_dependent_threshold": False,
    "after_spike_currents": True,
    "adapting_threshold": False
}

np3={
    "V_m": -79.0417277018229,
    "V_th": -49.63934810542196,
    "g": 4.4780284104908676,
    "E_L": -79.0417277018229,
    "C_m": 60.72689987399939,
    "t_ref": 1.4500000000000002,
    "V_reset": -79.0417277018229,
    "asc_init": [
        0.0,
        0.0
    ],
    "asc_decay": [
        0.029999999999999992,
        0.3
    ],
    "asc_amps": [
        -23.825265478178427,
        -292.06473034028727
    ],
    "tau_syn": [
        5.5,
        8.5,
        2.8,
        5.8
    ],
    "spike_dependent_threshold": False,
    "after_spike_currents": True,
    "adapting_threshold": False
}
################################
## Specify synapse properties ##
################################

delay = 1.5
# excitatory input to receptor_type 1
nest.CopyModel("static_synapse", "excitatory",
               {"weight": params['J'], "delay": delay, "receptor_type": 1})
# inhbitiory input to receptor_type 2 (makes weight automatically postive if negative weight is supplied)
nest.CopyModel("static_synapse", "inhibitory",
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

## Add populations to the network
network.addpop('glif_psc', params['N'][0], [neuron_params, np2, np3], pos_ex, nrec=800)
network.addpop('glif_psc', params['N'][1], [neuron_params, np2, np3], pos_in, nrec=200)

network.addpop('glif_psc', params['N'][0], [neuron_params, np2, np3], pos_ex, nrec=800)
network.addpop('glif_psc', params['N'][1], [neuron_params, np2, np3], pos_in, nrec=200)

network.addpop('glif_psc', params['N'][0], [neuron_params, np2, np3], pos_ex, nrec=800)
network.addpop('glif_psc', params['N'][1], [neuron_params, np2, np3], pos_in, nrec=200)

network.addpop('glif_psc', params['N'][0], [neuron_params, np2, np3], pos_ex, nrec=800)
network.addpop('glif_psc', params['N'][1], [neuron_params, np2, np3], pos_in, nrec=200)

network.addpop('glif_psc', params['N'][0], [neuron_params, np2, np3], pos_ex, nrec=800)
network.addpop('glif_psc', params['N'][1], [neuron_params, np2, np3], pos_in, nrec=200)

# add stimulation
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

## Define connectivity matrix
conn_matrix = np.array([[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
                        [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]])
syn_matrix = np.array([["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
                       ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
                       ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
                       ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
                       ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
                       ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
                       ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
                       ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
                       ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"],
                       ["excitatory", "inhibitory", "excitatory", "inhibitory","excitatory", "inhibitory", "excitatory", "inhibitory", "excitatory", "inhibitory"]])

## Connect all populations to each other according to the
## connectivity matrix and synaptic specifications
network.connect_all(conn_matrix, syn_matrix)

## simulate
network.simulate(params['sim_time'])

## Extract data from the network
mmdata, spikes = network.get_data()

## Plot spike data
raster(spikes, params['rec_start'], params['rec_stop'])
plt.show()

## Display the average firing rate in Hz
rate(spikes, params['rec_start'], params['rec_stop'])

## Approximate the lfp timecourse
#lfp_tc_l1, all_tc = approximate_lfp_timecourse(mmdata)
lfp_tc_l1 = approximate_lfp_timecourse(mmdata[0:2])
lfp_tc_l2 = approximate_lfp_timecourse(mmdata[2:4])
lfp_tc_l3 = approximate_lfp_timecourse(mmdata[4:6])
lfp_tc_l4 = approximate_lfp_timecourse(mmdata[6:8])
lfp_tc_l5 = approximate_lfp_timecourse(mmdata[8:10])


t = np.unique(mmdata[0]["times"])

# for l in all_tc:
#     for x in l:
#         plt.plot(t, x)

# plt.plot(t, lfp_tc, c='black', lw=5)
# plt.show()    

## plot the timecourse in the recorded time window
plt.plot(np.unique(mmdata[0]["times"]), lfp_tc_l1)
plt.plot(np.unique(mmdata[0]["times"]), lfp_tc_l2)
plt.plot(np.unique(mmdata[0]["times"]), lfp_tc_l3)
plt.plot(np.unique(mmdata[0]["times"]), lfp_tc_l4)
plt.plot(np.unique(mmdata[0]["times"]), lfp_tc_l5)
plt.show

newlst = np.array([lfp_tc_l1, lfp_tc_l2, lfp_tc_l3, lfp_tc_l4, lfp_tc_l5])

#icsd.CSD(lfp_tc)
