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

neuron_params_ex = {
    "V_m":            -70.,     ## Membrane potential in mV
    "V_th":           -50.,     ## instantaneous threshold in mV
#    "g":   np.array([2.08, 0.104, 0.327, 1.287]), ## Hopefully conductances of receptors
    "g_m":              25,     ## Membrance leak conductance in nS
    "E_L":            -70.,     ## Resting membrane potential in mV
    "C_m":             0.5,     ## Capacitance of membrane in picoFarad
    "t_ref":            1.,     ## Duration of refractory period in ms
    "V_reset":        -55.,     ## Reset potential of the membrane in mV (GLIF1 or GLIF3)
    "tau_minus":        2.,     ## Synapse decay time(??)
    "tau_syn":     (2., 10., 100., 2.),
    "E_rev":       (0.,0.,0.,0.)      ## Reversal potential
}

neuron_params_in = {
    "V_m":            -70.,     ## Membrane potential in mV
    "V_th":           -50.,     ## instantaneous threshold in mV
#    "g":   np.array([1.62, 0.081, 0.258, 1.002]), ## Hopefully conductances of receptors
    "g_m":              20,     ## Membrance leak conductance in nS
    "E_L":            -70.,     ## Resting membrane potential in mV
    "C_m":             0.2,     ## Capacitance of membrane in picoFarad
    "t_ref":            1.,     ## Duration of refractory period in ms
    "V_reset":        -55.,     ## Reset potential of the membrane in mV (GLIF1 or GLIF3)
    "tau_minus":        2.,     ## Synapse decay time(??)
    "tau_syn":         (2., 10., 100., 2.), ##AMPA/GABA/NMDA/NMDARISE
    "E_rev":          (-70.,-70.,-70.,-70.)      ## Reversal potential
}

params = {
    'num_neurons': 250,                # number of neurons in network
    'rho':  0.2,                        # fraction of inhibitory neurons
    'eps':  0.2,                        # probability to establish a connections
    'g':    5,                          # excitation-inhibition balance
    'eta':  2.5,                        # relative external rate
    'J_ex':    1.5,                        # postsynaptic amplitude in mV
    'J_in':    1.0,
    'neuron_params_ex': neuron_params_ex,     # single neuron parameters
    'neuron_params_in': neuron_params_in,
    'n_rec_ex':  100,                   # excitatory neurons to be recorded from
    'n_rec_in':  50,                   # inhibitory neurons to be recorded from
    'rec_start': 600.,                  # start point for recording spike trains
    'rec_stop':  800.,                   # end points for recording spike trains
    'nu_ext': 240.,                    # (neuron_params_ex['V_th']   # the external rate needs to be adapted to provide enough input (Brunel 2000)
    'nu_in': 250.,                                    # / (0.1 * 0.2*6000 * neuron_params_ex['tau_m']) ##J_ex*eps*num_ex
                                        # * 2.5 * 1000. * 0.2*6000), ## eta*1000*eps*num_ex
    'delay': 4.,                           ## connection specific delay
    'w_plus': 1.5,
    'w_i': 1.
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

#nu_th = neuron_params_ex['V_th'] / (params['J_ex'] * network.c_ex * neuron_params_ex['tau_m'])
#nu_ex = params['eta'] * nu_th

#ratio = nu_ex/nu_th



#print(f"Current Netork parameters: Nu_ex/Nu_th: {ratio}, Delay: 1.5ms, g: {params['g']}")
raster(test[0], test[1], params.get('rec_start'), params.get('rec_stop'))
rate(test[0], test[1], params.get('rec_start'), params.get('rec_stop'))
