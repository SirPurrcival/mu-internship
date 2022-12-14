# -*- coding: utf-8 -*-
"""
This is a heavily commented reference script.
Probably better as a notebook but I didn't feel
like opening jupyter notebooks today
"""
import nest
from matplotlib import pyplot as plt
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
glif_cond_params = {
    "V_m":            -70.,     ## Membrane potential in mV
    "V_th":         -51.68,     ## instantaneous threshold in mV
    "g_m":             9.4,     ## Membrance leak conductance in nS
    "E_L":          -78.85,     ## Resting membrane potential in mV
    "C_m":             0.5,     ## Capacitance of membrane in picoFarad
    "t_ref":          3.75,     ## Duration of refractory period in ms
    "V_reset":        -80.,     ## Reset potential of the membrane in mV (GLIF1 or GLIF3)
    "tau_minus":       20.      ## Synapse decay time(??)
    }

## Classic lif neuron
syn_tau = [2.0, 1.0]
n_lif = nest.Create("glif_psc",
                    params={"spike_dependent_threshold": False,
                            "after_spike_currents": False,
                            "adapting_threshold": False,
                            "tau_syn": syn_tau})


## If you ever feel like know what's going on in a neuron
## use nest.GetStatus()! Works with pretty much all things nest
nest.ResetKernel()
nest.SetKernelStatus({'local_num_threads': 4})  # Adapt if necessary

nest.print_time = True


neuronone = nest.Create("glif_cond", params=glif_cond_params)
neurontwo = nest.Create("glif_cond", params=glif_cond_params)
voltmeter = nest.Create("voltmeter")


nest.Connect(neuronone, neurontwo,
             syn_spec={'receptor_type': 2})

nest.GetStatus(neurontwo)
stimulus = nest.Create("poisson_generator")
stimulus.rate = 500
test_recorder = nest.Create("spike_recorder")

nest.Connect(neurontwo, test_recorder)


nest.Connect(stimulus, neurontwo,
             syn_spec={'receptor_type': 1,
                       'weight': 1.})
nest.Connect(voltmeter, neurontwo)
nest.Simulate(100)

nest.GetStatus(neurontwo)
print(nest.GetStatus(test_recorder))
nest.GetStatus(voltmeter)
nest.voltage_trace.from_device(voltmeter)
plt.show()
