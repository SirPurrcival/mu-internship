#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  5 09:23:10 2023

@author: meowlin
"""

import nest
import matplotlib.pyplot as plt

for vm in [-70, -65, -60, -56]:
    nest.ResetKernel()
    
    neuron = nest.Create("iaf_psc_exp", params={"V_m":vm})
    
    stimulus = nest.Create("poisson_generator")
    rec = nest.Create("multimeter", 
            params={'interval'   : 0.1,
            'start'      : 0,
            'label'      : 'multimeter',
            'record_from': ["V_m"]
            })
    
    nest.Connect(stimulus, neuron)
    nest.Connect(rec, neuron)
    
    
    for i in range(0,210000, 200):
        
        stimulus.rate = i
        nest.Simulate(1)
        
    data = nest.GetStatus(rec)[0]['events']
        
    plt.plot(data['times'], data['V_m'])

plt.show()

