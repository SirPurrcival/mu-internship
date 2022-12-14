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

nest.print_time = True
#nest.overwrite_files = True

network = Network(50000)
neuron_params_ex = {
    "V_m":            -70.,     ## Membrane potential in mV
    "V_th":           -50.,     ## instantaneous threshold in mV
#    "g":   np.array([2.08, 0.104, 0.327, 1.287]), ## Hopefully conductances of receptors
#    "g_m":              25,     ## Membrance leak conductance in nS
    "E_L":            -70.,     ## Resting membrane potential in mV
    "C_m":             0.5,     ## Capacitance of membrane in picoFarad
    "t_ref":            1.,     ## Duration of refractory period in ms
    "V_reset":        -55.,     ## Reset potential of the membrane in mV (GLIF1 or GLIF3)
    "tau_minus":        2.,     ## Synapse decay time(??)
#    "tau_syn":     (2., 10., 100., 2.),
#    "E_rev":       (0.,0.,0.,0.),      ## Reversal potential
    "spike_dependent_threshold": False,
    "after_spike_currents": False,
    "adapting_threshold": False,
    "tau_syn": [2.0, 1.0]
}


conn_spec={'rule': 'fixed_indegree', 'indegree': int(0.2*800)}
syn_spec_ex_ex={'synapse_model': 'stdp_synapse',
          'weight': 1.,
          'delay': 1., 
          'receptor_type': 1,
          'alpha': 0.5}
syn_spec_in_ex={'synapse_model': 'stdp_synapse',
          'weight': 1.,
          'delay': 1., 
          'receptor_type': 2,
          'alpha': 0.5}

## "glif_psc" doesn't seem to work with receptor type 3 and 4.
## Maybe synapse type is incorrect?
syn_spec_ex_in={'synapse_model': 'stdp_synapse',
          'weight': 1.,
          'delay': 1., 
          'receptor_type': 2,
          'alpha': 0.5}
syn_spec_in_in={'synapse_model': 'stdp_synapse',
          'weight': 1.,
          'delay': 1., 
          'receptor_type': 1,
          'alpha': 0.5}


network.addpop('glif_psc', 800,neuron_params_ex, record_from_pop=True, nrec=200)
network.addpop('glif_psc', 200,neuron_params_ex, record_from_pop=True, nrec=50)

conn_matrix = np.array([[0.2, 0.2],
                        [0.2,0.2]])
syn_matrix = np.array([[syn_spec_ex_ex, syn_spec_in_ex],
                      [syn_spec_ex_in, syn_spec_in_in]])


network.connect_all(network.get_pops(), conn_matrix, syn_matrix)

network.create()
network.simulate(1000)
test = network.get_data()
raster(test, 600, 800)
rate(test,600,800)
#plot_weight_matrices(network.get_pops()[0], network.get_pops()[1])
