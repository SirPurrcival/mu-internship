#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  9 10:27:43 2023

@author: meowlin
"""

import nest
import matplotlib.pyplot as plt
import numpy as np
import math

ar = np.arange(0.25, 1, 0.1)
for syn_ex in ar:

    nest.ResetKernel()
    rec = nest.Create("multimeter", params={'record_from': ["V_m"]})
    neuron = nest.Create("iaf_psc_exp", 1,  params={'tau_syn_in': syn_ex})
    stimulus = nest.Create("poisson_generator")
    stimulus.rate = 1000000.
    
    nest.Connect(stimulus, neuron)
    nest.Connect(rec, neuron)
    
    nest.Simulate(40)
    
    data = nest.GetStatus(rec)
    
    plt.plot(data[0]['events']['times'], data[0]['events']['V_m'])
plt.legend(ar)

meow = []
k=1.5
time = np.arange(0,5,0.1)
for i in time:
    meow.append(math.exp(-k*i))
plt.plot(time, meow)
    
    
    