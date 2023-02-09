import numpy as np
import pylab as plt

from new_functions import approximate_shitty_lfp

import nest

nest.ResetKernel()
resolution = 0.1
nest.resolution = resolution
nest.local_num_threads = 5

M = 2               # number of populations
NE = 800            # number of excitatory neurons
NI = 200            # number of inhibitory neurons
P = 0.1             # connection probability
CE = int(NE * P)    # number of excitatory recurrent synapses
CI = int(NI * P)    # number of inhibitory recurrent syanpses
J = 0.1             # synaptic weight of single synapse
g = NE / NI         # excitatory-inhibitory ratio
J_ex = J            # excitatory weight of single synapse
J_in = J * g        # inhibitory weight of single synapse

# setup neurons
E_lif = nest.Create("glif_psc", NE,
                    params={
    "V_m": -79.0417277018229,
    "V_th": -49.63934810542196,
    "g": 3.4780284104908676,
    "E_L": -79.0417277018229,
    "C_m": 60.72689987399939,
    "t_ref": 1.4500000000000002,
    "V_reset": -79.0417277018229,
    "asc_init": [
        0.0,
        0.0
    ],
    "asc_decay": [
        0.029999999999999992,
        0.3
    ],
    "asc_amps": [
        -23.825265478178427,
        -292.06473034028727
    ],
    "tau_syn": [
        5.5,
        8.5,
        2.8,
        5.8
    ],
    "spike_dependent_threshold": False,
    "after_spike_currents": True,
    "adapting_threshold": False
})
I_lif = nest.Create("glif_psc", NI,
                    params={
    "V_m": -79.0417277018229,
    "V_th": -49.63934810542196,
    "g": 3.4780284104908676,
    "E_L": -79.0417277018229,
    "C_m": 60.72689987399939,
    "t_ref": 1.4500000000000002,
    "V_reset": -79.0417277018229,
    "asc_init": [
        0.0,
        0.0
    ],
    "asc_decay": [
        0.029999999999999992,
        0.3
    ],
    "asc_amps": [
        -23.825265478178427,
        -292.06473034028727
    ],
    "tau_syn": [
        5.5,
        8.5,
        2.8,
        5.8
    ],
    "spike_dependent_threshold": False,
    "after_spike_currents": True,
    "adapting_threshold": False
})
populations = [E_lif, I_lif]

# connect populations
delay = 1.5
# excitatory input to receptor_type 1
nest.CopyModel("static_synapse", "excitatory",
               {"weight": J_ex, "delay": delay, "receptor_type": 1})
# inhbitiory input to receptor_type 2 (makes weight automatically postive if negative weight is supplied)
nest.CopyModel("static_synapse", "inhibitory",
               {"weight": J_in, "delay": delay, "receptor_type": 2})

conn_E = {"rule": "fixed_indegree", "indegree": CE}
nest.Connect(E_lif, E_lif + I_lif, conn_E, "excitatory")
conn_I = {"rule": "fixed_indegree", "indegree": CI}
nest.Connect(I_lif, E_lif + I_lif, conn_I, "inhibitory")

# add input
nu_bg = 8       # background rate
C_bg  = 850     # number of background connections
pg = nest.Create("poisson_generator",
                 params={"rate": nu_bg * C_bg})
nest.Connect(pg, populations[0], syn_spec={"delay": 1.5, "receptor_type": 1})
nest.Connect(pg, populations[1], syn_spec={"delay": 1.5, "receptor_type": 1})


# setup recording devices
mm = nest.Create("multimeter", M,
                 params={"interval": resolution,
                         "record_from": ["V_m", "I_syn"]})
nest.Connect(mm[0], E_lif)
nest.Connect(mm[1], I_lif)

sr = nest.Create("spike_recorder", M)
nest.Connect(E_lif, sr[0])
nest.Connect(I_lif, sr[1])

# simulate
nest.Simulate(1000)
print("done!")

# get spike data
senders = np.empty(M, dtype=object)
times   = np.empty(M, dtype=object)
for m in range(M):
    spike_data = sr[m].events
    senders[m] = spike_data["senders"]
    times[m] = spike_data["times"]

colors = ['blue', 'red']
plt.figure()
for m in range(M):
    plt.scatter(times[m], senders[m], s=1, color=colors[m])
plt.gca().invert_yaxis()
plt.show()
