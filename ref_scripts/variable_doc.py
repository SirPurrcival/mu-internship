# -*- coding: utf-8 -*-
"""
This is a heavily commented reference script.
Probably better as a notebook but I didn't feel
like opening jupyter notebooks today
"""

iaf_psc_delta_params = {"C_m":     0.5, #0.5     ## capacity of membrane
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

## Model documentation: https://nest-simulator.readthedocs.io/en/stable/models/glif_psc.html
glif_psc_params = {
    "V_m":          -78.85,     ## Membrane potential in mV
    "V_th":         -51.68,     ## instantaneous threshold in mV
    "g":              9.43,     ## Membrance conduction in nS
    "E_L":          -78.85,     ## Resting membrane potential in mV
    "C_m":          -58.72,     ## Capacitance of membrane in picoFarad
    "t_ref":          3.75,     ## Duration of refractory period in ms
    "V_reset":       -78.85     ## Reset potential of the membrane in mV (GLIF1 or GLIF3)
    }



## If you ever feel like know what's going on in a neuron
## use nest.GetStatus()! Works with pretty much all things nest
import nest
nya = nest.Create("glif_psc")
nest.GetStatus(nya)
