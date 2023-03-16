import numpy as np
import pylab as plt
from functions import get_firing_rate, get_irregularity, get_synchrony

import IPython

inpute_net1 = 9
inpute_net2 = 12
inputi_net1 = 4
inputi_net2 = 4
cop_val    = 0.0


Ne1 = 200
Ne2 = 200
Ni1 = 50
Ni2 = 50
Ne  = Ne1 + Ne2
Ni  = Ni1 + Ni2

a = np.concatenate([0.02 * np.ones(Ne), 0.1*np.ones(Ni)]) # Time scale of recovery variable
b = np.concatenate([0.2  * np.ones(Ne), 0.2*np.ones(Ni)]) # Sensitivity of recovery variable
c = np.concatenate([-65  * np.ones(Ne), -65*np.ones(Ni)]) # Reset value of membrane potential
d = np.concatenate([8    * np.ones(Ne), 2  *np.ones(Ni)]) # Reset value of recovery variable
v = -65 * np.ones(Ne + Ni) # Initial values of membrane potential
u = b * v   # Initial values of recovery variable
spikes2 = [[]]*(Ne+Ni) # Spike timings
spikes = []

simulation_time = 1000
dt = 1
Ntot = Ne+Ni

# Gaussian input
mean_E = [inpute_net1, inpute_net2]
var_E  = 3
mean_I = [inputi_net1, inputi_net2]
var_I  = 3

gampa = np.zeros(Ne)
gaba  = np.zeros(Ni)
decay_ampa = 1
decay_gaba = 7
rise_ampa  = 0.15
rise_gaba  = 0.2

# Connectivity
EE = 0.05
EI = 0.4
IE = 0.3
II = 0.2

# E -> I
C1 = cop_val    # E1 > I2
C2 = cop_val    # E2 > I1
# E -> E
C3 = cop_val/4  # E1 > E2
C4 = cop_val/4  # E2 > E1

S = np.zeros((Ntot, Ntot))

E1_idx = np.arange(Ne1)
E2_idx = np.arange(Ne1, Ne1+Ne2)
I1_idx = np.arange(Ne1+Ne2, Ne1+Ne2+Ni1)
I2_idx = np.arange(Ne1+Ne2+Ni1, Ntot)

# Within network
S[np.ix_(E1_idx, E1_idx)] = EE * np.random.randn(Ne1, Ne1)
S[np.ix_(I1_idx, E1_idx)] = EI * np.random.randn(Ni1, Ne1)
S[np.ix_(E1_idx, I1_idx)] = IE * np.random.randn(Ne1, Ni1)
S[np.ix_(I1_idx, I1_idx)] = II * np.random.randn(Ni1, Ni1)

S[np.ix_(E2_idx, E2_idx)] = EE * np.random.randn(Ne2, Ne2)
S[np.ix_(I2_idx, E2_idx)] = EI * np.random.randn(Ni2, Ne2)
S[np.ix_(E2_idx, I2_idx)] = IE * np.random.randn(Ne2, Ni2)
S[np.ix_(I2_idx, I2_idx)] = II * np.random.randn(Ni2, Ni2)

# Between network
S[np.ix_(I2_idx, E1_idx)] = C1 * np.random.randn(Ni2, Ne1)
S[np.ix_(I1_idx, E2_idx)] = C2 * np.random.randn(Ni1, Ne2)
S[np.ix_(E2_idx, E1_idx)] = C3 * np.random.randn(Ne2, Ne1)
S[np.ix_(E1_idx, E2_idx)] = C4 * np.random.randn(Ne1, Ne2)

V = np.zeros((Ntot, int(simulation_time/dt)))
U = np.zeros((Ntot, int(simulation_time/dt)))
F = np.zeros((Ntot, int(simulation_time/dt)))
# Simulation
for t in range(int(simulation_time/dt)):
    I = np.concatenate([np.random.normal(mean_E[0], var_E, Ne1), 
                        np.random.normal(mean_E[1], var_I, Ne2),
                        np.random.normal(mean_I[0], var_E, Ni1),
                        np.random.normal(mean_I[1], var_I, Ni2)])   # Thalamic input
    
    fired = np.where(v >= 30)[0] # indices of spikes
    
    for spike in fired:
        if t == 3:
            print(f"appending {t} in {spike}")
        spikes2[spike].append(t)
    
    
    spikes.append(fired)
    v[fired] = c[fired]
    u[fired] = u[fired] + d[fired]
    dgampa = 0.3 * (((1 + np.tanh((v[:Ne]/10) + 2))/2) * (1-gampa)/rise_ampa - gampa/decay_ampa)
    gampa = gampa + dt * dgampa 
    dgaba = 0.3 * (((1 + np.tanh((v[Ne:]/10) + 2))/2) * (1-gaba)/rise_gaba - gaba/decay_gaba)
    gaba = gaba + dt * dgaba
    
    I = I + np.dot(S, np.concatenate([gampa, gaba]))
    v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)   # step 0.5 ms
    v = v + 0.5 * (0.04 * v**2 + 5 * v + 140 - u + I)   # for numerical
    u = u + a * (b * v - u)                             # stability

    V[:, t] = v
    U[:, t] = u
    F[fired, t] = 1

# Post-processing
plt.figure()
plt.subplot(311)
plt.title('Membrane potential')
plt.plot(np.mean(V[E1_idx, :], axis=0), color='blue', linewidth=2, linestyle='-',  label='E1')
plt.plot(np.mean(V[E2_idx, :], axis=0), color='blue', linewidth=2, linestyle='--', label='E2')
plt.plot(np.mean(V[I1_idx, :], axis=0), color='red',  linewidth=2, linestyle='-',  label='I1')
plt.plot(np.mean(V[I2_idx, :], axis=0), color='red',  linewidth=2, linestyle='--', label='I2')
plt.legend()

plt.subplot(312)
plt.title('Recovery variable')
plt.plot(np.mean(U[E1_idx, :], axis=0), color='blue', linewidth=2, linestyle='-',  label='E1')
plt.plot(np.mean(U[E2_idx, :], axis=0), color='blue', linewidth=2, linestyle='--', label='E2')
plt.plot(np.mean(U[I1_idx, :], axis=0), color='red',  linewidth=2, linestyle='-',  label='I1')
plt.plot(np.mean(U[I2_idx, :], axis=0), color='red',  linewidth=2, linestyle='--', label='I2')

plt.subplot(313)
plt.title('Spikes')
plt.imshow(F, aspect='auto', cmap='coolwarm_r')

plt.show()

IPython.embed()