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
from new_functions import Network, raster, rate

nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads': 4})  # Adapt if necessary

nest.print_time = False
#nest.overwrite_files = True



# neuron_params_ex = {"C_m":   0.5,            ## capacity of membrane
#                   "tau_m":   20.,            ## membrane time constant
#                   "t_ref":   2.0,            ## duration of refractory period
#                   "E_L":     0.0,            ## resting membrane potential
#                   "V_m":     0.0,            ## membrane potential
#                   "V_th":    20.,            ## spike threshold
#                   "V_reset": 0.0             ## Reset potential of membrane
#                   }

params = {
    'num_neurons': 800,                # number of neurons in network
    'rho':  0.2,                        # fraction of inhibitory neurons
    'eps':  0.2,                        # probability to establish a connections
    'g':    5,                          # excitation-inhibition balance
    'eta':  4.4,                          # relative external rate
    'J':    0.1,                        # postsynaptic amplitude in mV
    'n_rec_ex':  6000,                   # excitatory neurons to be recorded from
    'n_rec_in':  2000,                   # inhibitory neurons to be recorded from
    'rec_start': 600.,                  # start point for recording spike trains
    'rec_stop':  800.                   # end points for recording spike trains
    }

neuron_params_ex = {
    #"C_m":     1.0,                     ## capacity of membrane
    "t_ref":   2.0,                     ## duration of refractory period
    "E_L":     0.0,                     ## resting membrane potential
    "V_m":     0.0,                     ## membrane potential
    "V_th":    30.,                     ## spike threshold
    "V_reset": 0.0,                     ## Reset potential of membrane
    "spike_dependent_threshold": False,
    "after_spike_currents": False,
    "adapting_threshold": False,
    "tau_syn": [2, 20]
}

syn_spec_ex_ex={
    'weight': params['J'],
    'delay': 1.5, 
    'receptor_type': 1
    }
syn_spec_in_ex={
    'weight': params['J']*-params['g'],
    'delay': 1.5, 
    'receptor_type': 2
     }
syn_spec_ex_in={#'synapse_model': 'stdp_synapse',
    'weight': params['J'],
    'delay': 1.5,
    'receptor_type': 1
    }
syn_spec_in_in={#'synapse_model': 'stdp_synapse',
    'weight': params['J']*-params['g'],
    'delay': 1.5, 
    'receptor_type': 2
    }



## V_th / (excitatory weight * number of exc. conns * tau_m) * rel. ext rate * 1000 * number of exc. conns
ext_rate = (neuron_params_ex['V_th']   # the external rate needs to be adapted to provide enough input (Brunel 2000)
                  / (params['J'] * params['num_neurons'] * params['eps'] * neuron_params_ex['tau_syn'][0]) #neuron_params_ex['tau_syn'])
                  * params['eta'] * 1000. * params['num_neurons'] * params['eps'])


nu_th = neuron_params_ex['V_th'] / (params['J'] * params['num_neurons'] * params['eps'] * neuron_params_ex['tau_syn'][0])
nu_ex = params['eta'] * nu_th

ratio = nu_ex/nu_th

network = Network(ext_rate)
network.addpop('glif_psc', int(params['num_neurons']*(1-params['rho'])),neuron_params_ex, record_from_pop=True, nrec=400)
network.addpop('glif_psc', int(params['num_neurons']*params['rho']),neuron_params_ex, record_from_pop=True, nrec=100)

# network.addpop('iaf_psc_delta', int(params['num_neurons']*(1-params['rho'])),neuron_params_ex, record_from_pop=True, nrec=200)
# network.addpop('iaf_psc_delta', int(params['num_neurons']*params['rho']),neuron_params_ex, record_from_pop=True, nrec=50)



conn_matrix = np.array([[params['eps']*(1-params['rho']), params['eps']],
                        [params['eps']*(1-params['rho']), 0.0]])
syn_matrix = np.array([[syn_spec_ex_ex, syn_spec_in_ex],
                      [syn_spec_ex_in, syn_spec_in_in]])


network.connect_all(network.get_pops(), conn_matrix, syn_matrix)

network.create()
network.simulate(1000)
test = network.get_data()
raster(test, 500, 700)
rate(test,600,800)
