import numpy as np
import pylab as plt

import nest

import IPython

from plotPowerSpectrum import plotPowerSpectrum


nest.ResetKernel()
resolution = 0.05
nest.resolution = resolution

# setup neurons
n_lif = nest.Create("glif_cond",
                    params={"spike_dependent_threshold": False,
                            "after_spike_currents":      False,
                            "adapting_threshold":        False})

# add input
espikes = nest.Create("spike_generator",
                      params={"spike_times":    [10., 100., 150.],
                              "spike_weights":  [20.] * 3})
ispikes = nest.Create("spike_generator",
                      params={"spike_times":    [10., 100., 150.],
                              "spike_weights":  [20.] * 3})                            
nest.Connect(espikes, n_lif,
             syn_spec={"delay": resolution, "receptor_type": 1})
nest.Connect(ispikes, n_lif,
             syn_spec={"delay": resolution, "receptor_type": 2})

# excitatory input to receptor_type 1
# inhbitiory input to receptor_type 2
pg = nest.Create("poisson_generator",
                 params={"rate": 15000})
pn = nest.Create("parrot_neuron")
nest.Connect(pg, pn, syn_spec={"delay": resolution})
nest.Connect(pn, n_lif, syn_spec={"delay": resolution, "receptor_type": 1})



# setup recording devices
mm = nest.Create("multimeter",
                 params={"interval": resolution,
                         "record_from": ["V_m", "I", "g_1", "g_2",
                         "threshold",
                         "threshold_spike",
                         "threshold_voltage",
                         "ASCurrents_sum"]})
nest.Connect(mm, n_lif)

sr = nest.Create("spike_recorder")
nest.Connect(n_lif, sr)

# simulate
nest.Simulate(1000)

data = mm.events
senders = data['senders']

spike_data = sr.events
spike_senders = spike_data["senders"]
spikes = spike_data["times"]

plt.figure()
t = data["times"][senders == 1]
plt.plot(t, data["V_m"][senders == 1], "black")
plt.show()

