#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 16:36:41 2023

@author: fiona
"""

import numpy as np
import pylab as plt

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

# Connectivity matrix layertype X layertype
C = np.array([[0.656, 0.356, 0.093, 0.068, 0.4644, 0.148, 0, 0, 0, 0.148, 0, 0, 0.148, 0, 0, 0],
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