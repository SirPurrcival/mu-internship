#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  8 12:22:23 2022

@author: meowlin
"""

# Import libraries
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nest
import itertools

from functions import Network, raster, rate

neuron_params_ex = {"C_m":     0.5, #0.5     ## capacity of membrane
                  "tau_m":   20.,            ## membrane time constant
                  "t_ref":   2.0,            ## duration of refractory period
                  "E_L":     0.0,            ## resting membrane potential
                  "V_m":     0.0,            ## membrane potential
                  "V_th":    20.,            ## spike threshold
                  "V_reset": 0.0,            ## Reset potential of membrane
                  "I_e":     0.0,            ## constant input current
                  "V_min":   -1.7976931348623157e+308, ## absolute lower value  for the membrane potential
                  "refractory_input": False  ## if true, do not discard input during refractory period
                  }
neuron_params_in = {"C_m":     0.2, #0.2
                  "tau_m":   20.,
                  "t_ref":   1.0,#1.0
                  "E_L":     0.0,
                  "V_reset": 0.0,
                  "V_m":     0.0,
                  "V_th":    20.
                  }

params = {
    'num_neurons': 10000,                # number of neurons in network
    'rho':  0.2,                        # fraction of inhibitory neurons
    'eps':  0.2,                        # probability to establish a connections
    'g':    5,                          # excitation-inhibition balance
    'eta':  2.5,                        # relative external rate
    'J_ex':    0.1,                        # postsynaptic amplitude in mV
    'J_in':    -1.0,
    'neuron_params_ex': neuron_params_ex,     # single neuron parameters
    'neuron_params_in': neuron_params_in,
    'n_rec_ex':  600,                   # excitatory neurons to be recorded from
    'n_rec_in':  200,                   # inhibitory neurons to be recorded from
    'rec_start': 600.,                  # start point for recording spike trains
    'rec_stop':  800.                   # end points for recording spike trains
    }

nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads': 4})  # Adapt if necessary

nest.print_time = True
#nest.overwrite_files = True

network = Network(**params)
network.create()
network.simulate(1000)
test = network.get_data()

# nya = test[2]
# for a in range(1,601):
#     counter = 0
#     for i in nya:
#         if i == a:
#             counter += 1
#     print("Neuron " + str(a) + " fired " + str(counter) + " times during this simulation.")
    

# print("Neuron 1 fired " + str(counter) + " times during this simulation.")
#nu_th = theta / (J * CE * tauMem)
#nu_ex = eta * nu_th

#theta = V_th

nu_th = neuron_params_ex['V_th'] / (params['J_ex'] * network.c_ex * neuron_params_ex['tau_m'])
nu_ex = params['eta'] * nu_th

ratio = nu_ex/nu_th


print(f"Current Netork parameters: Nu_ex/Nu_th: {ratio}, Delay: 1.5ms, g: {params['g']}")
raster(test[0], test[1], params.get('rec_start'), params.get('rec_stop'))
#nest.raster_plot.from_device(network.spike_recorder_ex)
rate(test[0], test[1], params.get('rec_start'), params.get('rec_stop'))

# testneuron = nest.Create('iaf_psc_delta')
# nest.GetStatus(testneuron)
