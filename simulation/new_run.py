#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 12:15:13 2022

@author: meowlin
"""

import numpy as np
import matplotlib.pyplot as plt
import nest
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from new_functions import Network, raster, rate, approximate_lfp

nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads': 4})  # Adapt if necessary

nest.print_time = False
#nest.overwrite_files = True
resolution = 0.1
nest.resolution = resolution

params = {
    'N':    [800, 200],                 # number of neurons in network
    'rho':  0.2,                        # fraction of inhibitory neurons
    'eps':  0.1,                        # probability to establish a connections
    'g':    4,                          # excitation-inhibition balance
    'eta':  2,                          # relative external rate
    'J':    0.1,                        # postsynaptic amplitude in mV
    'n_rec_ex':  800,                   # excitatory neurons to be recorded from
    'n_rec_in':  200,                   # inhibitory neurons to be recorded from
    'rec_start': 600.,                  # start point for recording spike trains
    'rec_stop':  800.                   # end points for recording spike trains
    }

neuron_params = {
    "C_m":     1.0,                     ## capacity of membrane
    "t_ref":   2.0,                     ## duration of refractory period
    "E_L":     0.0,                     ## resting membrane potential
    "V_m":     0.0,                     ## membrane potential
    "V_th":    20.,                     ## spike threshold
    "V_reset": 10.,                     ## Reset potential of membrane
    "spike_dependent_threshold": False,
    "after_spike_currents": False,
    "adapting_threshold": False,
}

syn_spec_ex_ex = {
    'synapse_model': 'static_synapse',
    'weight': params['J'],
    'delay': 1.5,
    'receptor_type': 1
    }
syn_spec_in_ex = {
    'synapse_model': 'static_synapse',
    'weight': params['J']*-params['g'],
    'delay': 1.5,
    'receptor_type': 1
     }
syn_spec_ex_in = {
    'synapse_model': 'static_synapse',
    'weight': params['J'],
    'delay': 1.5,
    'receptor_type': 1
    }
syn_spec_in_in = {
    'synapse_model': 'static_synapse',
    'weight': params['J']*-params['g'],
    'delay': 1.5,
    'receptor_type': 1
    }



## From the old script:
# ext_rate = (neuron_params['V_th']   # the external rate needs to be adapted to provide enough input (Brunel 2000)
#             / (params['J'] * params['N'][0] * params['eps'] * neuron_params['tau_syn'][0])
#             * params['eta'] * 1000. * params['N'][0] * params['eps'])
ext_rate = 33.5e4


network = Network(resolution)

## Positions
pos_ex = nest.spatial.free(nest.random.normal(mean=0.5, std=1.),
                        num_dimensions=3)
pos_in = nest.spatial.free(nest.random.normal(mean=0., std=1.),
                        num_dimensions=3)

network.addpop('glif_psc', params['N'][0], neuron_params, pos_ex, record_from_pop=True, nrec=800)
network.addpop('glif_psc', params['N'][1], neuron_params, pos_in, record_from_pop=True, nrec=200)

# add stimulation
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=0) # to excitatory population
network.add_stimulation(source={'type': 'poisson_generator', 'rate': ext_rate}, target=1) # to inhibitory population

## Define connectivity matrix
conn_matrix = np.array([[params['eps'], params['eps']],
                        [params['eps'], params['eps']]])
syn_matrix = np.array([[syn_spec_ex_ex, syn_spec_in_ex],
                       [syn_spec_ex_in, syn_spec_in_in]])


network.connect_all(conn_matrix, syn_matrix)

simtime=1000

network.simulate(simtime)
spikes, mmdata = network.get_data()
raster(spikes, params['rec_start'], params['rec_stop'])
plt.show()

## TODO: This one still needs a small rewrite. Currently it takes the average of all spikes over the simulation time, which is only Hz if
##the simulation time is 1000ms. 
rate(spikes, params['rec_start'], params['rec_stop'])

spatial_data = [network.get_pops()[0].spatial, network.get_pops()[1].spatial] 

meow, cex, cin = approximate_lfp(resolution, mmdata, simtime, spatial_data)

plt.plot(meow)
#plt.xlim([0,200])
plt.show()
nya = network.get_pops()[0]
nya.spatial

for pop in network.get_pops():
    nest.PlotLayer(pop)
