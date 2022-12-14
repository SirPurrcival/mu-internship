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

network = Network()
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

conn_matrix = np.array([[0.2, 0.2],[0.2,0.2]])
syn_matrix = np.array([[syn_spec_ex_ex, syn_spec_in_ex],
                      [syn_spec_ex_in, syn_spec_in_in]])


network.connect_all(network.get_pops(), conn_matrix, syn_matrix)

network.create()
network.simulate(1000)
test = network.get_data()


def plot_weight_matrices(E_neurons, I_neurons):

    W_EE = np.zeros([len(E_neurons), len(E_neurons)])
    W_EI = np.zeros([len(I_neurons), len(E_neurons)])
    W_IE = np.zeros([len(E_neurons), len(I_neurons)])
    W_II = np.zeros([len(I_neurons), len(I_neurons)])

    a_EE = nest.GetConnections(E_neurons, E_neurons)

    # We extract the value of the connection weight for all the connections between these populations
    c_EE = a_EE.weight

    # Repeat the two previous steps for all other connection types
    a_EI = nest.GetConnections(I_neurons, E_neurons)
    c_EI = a_EI.weight
    a_IE = nest.GetConnections(E_neurons, I_neurons)
    c_IE = a_IE.weight
    a_II = nest.GetConnections(I_neurons, I_neurons)
    c_II = a_II.weight

    # We now iterate through the range of all connections of each type.
    # To populate the corresponding weight matrix, we begin by identifying
    # the source-node_id (by using .source) and the target-node_id.
    # For each node_id, we subtract the minimum node_id within the corresponding
    # population, to assure the matrix indices range from 0 to the size of
    # the population.

    # After determining the matrix indices [i, j], for each connection
    # object, the corresponding weight is added to the entry W[i,j].
    # The procedure is then repeated for all the different connection types.
    a_EE_src = a_EE.source
    a_EE_trg = a_EE.target
    a_EI_src = a_EI.source
    a_EI_trg = a_EI.target
    a_IE_src = a_IE.source
    a_IE_trg = a_IE.target
    a_II_src = a_II.source
    a_II_trg = a_II.target

    min_E = min(E_neurons.tolist())
    min_I = min(I_neurons.tolist())

    for idx in range(len(a_EE)):
        W_EE[a_EE_src[idx] - min_E, a_EE_trg[idx] - min_E] += c_EE[idx]
    for idx in range(len(a_EI)):
        W_EI[a_EI_src[idx] - min_I, a_EI_trg[idx] - min_E] += c_EI[idx]
    for idx in range(len(a_IE)):
        W_IE[a_IE_src[idx] - min_E, a_IE_trg[idx] - min_I] += c_IE[idx]
    for idx in range(len(a_II)):
        W_II[a_II_src[idx] - min_I, a_II_trg[idx] - min_I] += c_II[idx]

    fig = plt.figure()
    fig.suptitle('Weight matrices', fontsize=14)
    gs = gridspec.GridSpec(4, 4)
    ax1 = plt.subplot(gs[:-1, :-1])
    ax2 = plt.subplot(gs[:-1, -1])
    ax3 = plt.subplot(gs[-1, :-1])
    ax4 = plt.subplot(gs[-1, -1])

    plt1 = ax1.imshow(W_EE, cmap='jet')

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(plt1, cax=cax)

    ax1.set_title('$W_{EE}$')
    plt.tight_layout()

    plt2 = ax2.imshow(W_IE)
    plt2.set_cmap('jet')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(plt2, cax=cax)
    ax2.set_title('$W_{EI}$')
    plt.tight_layout()

    plt3 = ax3.imshow(W_EI)
    plt3.set_cmap('jet')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(plt3, cax=cax)
    ax3.set_title('$W_{IE}$')
    plt.tight_layout()

    plt4 = ax4.imshow(W_II)
    plt4.set_cmap('jet')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("right", "5%", pad="3%")
    plt.colorbar(plt4, cax=cax)
    ax4.set_title('$W_{II}$')
    plt.tight_layout()

#plot_weight_matrices(network.get_pops()[0], network.get_pops()[1])
