# Import libraries
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nest
import itertools

## 

# Create classes
class Network():
    def __init__(self, num_neurons, rho, eps, g, eta, J_ex, J_in, neuron_params_ex, neuron_params_in, n_rec_ex, n_rec_in, rec_start, rec_stop):
        self.num_neurons = num_neurons
        self.num_ex = int((1 - rho) * num_neurons)  # number of excitatory neurons
        self.num_in = int(rho * num_neurons)        # number of inhibitory neurons
        self.c_ex = int(eps * self.num_ex)          # number of excitatory connections
        self.c_in = int(eps * self.num_in)          # number of inhibitory connections
        self.J_ex = J_ex                               # excitatory weight
        self.J_in = J_in #-g*J                            # inhibitory weight
        self.n_rec_ex = n_rec_ex                    # number of recorded excitatory neurons
        self.n_rec_in = n_rec_in                    # number of recorded inhibitory neurons
        self.rec_start = rec_start
        self.rec_stop = rec_stop
        self.neuron_params_ex = neuron_params_ex          # neuron params
        self.neuron_params_in = neuron_params_in
        self.ext_rate = (self.neuron_params_ex['V_th']   # the external rate needs to be adapted to provide enough input (Brunel 2000)
                         / (J_ex * self.c_ex * self.neuron_params_ex['tau_m'])
                         * eta * 1000. * self.c_ex)

    def create(self):

        # Create the network

        # First create the neurons - the excitatory and inhibitory populations
        ## Your code here
        self.neurons_ex = nest.Create('iaf_psc_delta', self.num_ex, params=self.neuron_params_ex)
        self.neurons_in = nest.Create('iaf_psc_delta', self.num_in, params=self.neuron_params_in)
        self.neurons = self.neurons_ex + self.neurons_in
        #self.neurons_ex = self.neurons[:self.num_ex]
        #self.neurons_in = self.neurons[self.num_ex:]

        # Then create the external poisson spike generator
        ## Your code here
        
        self.p_noise = nest.Create("poisson_generator")
        self.p_noise.rate = self.ext_rate
        #print("Tself.ext_rate" + str(self.ext_rate))

        # Then create spike detectors
        # (disclaimer: dependening on NEST version, the device might be 'spike_detector' or 'spike_recorder'
        ## Your code here
        self.spike_recorder_ex = nest.Create("spike_recorder", 1)
        self.spike_recorder_in = nest.Create("spike_recorder", 1)
        # Next we connect the excitatory and inhibitory neurons to each other, choose a delay of 1.5 ms
        nest.Connect(self.neurons_ex, self.neurons,
             conn_spec={'rule': 'fixed_indegree', 'indegree': self.c_ex},
             syn_spec={'weight': self.J_ex, 'delay': 1.5})
        # Now also connect the inhibitory neurons to the other neurons
        ## Your code here
        nest.Connect(self.neurons_in, self.neurons,
                     conn_spec={'rule': 'fixed_indegree', 'indegree': self.c_in},
                     syn_spec={'weight': self.J_in, 'delay': 1.5})

        # Then we connect the external drive to the neurons with weight J_ex
        ## Your code here
        nest.Connect(self.p_noise, self.neurons,
                     syn_spec={'weight': self.J_ex})

        # Then we connect the the neurons to the spike detectors
        # Note: You can use slicing for nest node collections as well
        ## Your code here
        nest.Connect(self.neurons_ex[:self.n_rec_ex], self.spike_recorder_ex)
        nest.Connect(self.neurons_in[:self.n_rec_in], self.spike_recorder_in)

    def simulate(self, t_sim):
        # Simulate the network with specified
        nest.Simulate(t_sim)

    def get_data(self):
        # Define lists to store spike trains in
        # self.spikes_ex = []
        # self.spikes_in = []
        
        self.spike_times_ex = nest.GetStatus(self.spike_recorder_ex)
        self.spike_times_in = nest.GetStatus(self.spike_recorder_in)
        

        # There are several ways in which you can obtain the data recorded by the spikerecorders
        # One example is given below.
        # You can get the recorded quantities from the spike recorder with nest.GetStatus
        # You may loop over the entries of the GetStatus return
        # you might want to sort the spike times, they are not by default
        ## Your code here
        
        
        self.idx_ex = np.argsort(self.spike_times_ex[0]['events']['senders'])
        self.sorted_sx = self.spike_times_ex[0]['events']['senders'][self.idx_ex]
        self.tmp_ex = self.spike_times_ex[0]['events']['times'][self.idx_ex]

        self.spikes_ex = np.split(self.tmp_ex, np.where(np.diff(self.sorted_sx) > 0)[0] + 1)
        #for a in self.spikes_ex:
        #    a.sort()
        
        self.idx_in = np.argsort(self.spike_times_in[0]['events']['senders'])
        self.sorted_si = self.spike_times_in[0]['events']['senders'][self.idx_in]
        self.tmp_in = self.spike_times_in[0]['events']['times'][self.idx_in]

        self.spikes_in = np.split(self.tmp_in, np.where(np.diff(self.sorted_si) > 0)[0] + 1)
        #for b in self.spikes_in:
        #    b.sort()
        #for item in self.spike_times_ex:
        #    self.spikes_ex.append(np.sort(item['events']['times']))
        # 
        #for item in self.spike_times_in:
        #    self.spikes_in.append(np.sort(item['events']['times'])) #['times']
        
        
        # hint: another option would be to obtain both the times and the senders (neurons).
        # This way you obtain information about which neuron spiked at which time.
        # e.g. senders = nest.GetStatus(self.spikes_recorder, 'events')[0]['senders']
        #      times   = nest.GetStatus(self.spikes_recorder, 'events')[0]['times']
        # Try to practice with the nest.GetStatus command.
        
        return self.spikes_ex, self.spikes_in, self.sorted_sx

# Helper functions
def raster(spikes_ex, spikes_in, rec_start, rec_stop, figsize=(9, 5)):

    spikes_ex_total = list(itertools.chain(*spikes_ex))
    spikes_in_total = list(itertools.chain(*spikes_in))
    spikes_total = spikes_ex_total + spikes_in_total

    n_rec_ex = len(spikes_ex)
    n_rec_in = len(spikes_in)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 1)

    ax1 = fig.add_subplot(gs[:4,0])
    ax2 = fig.add_subplot(gs[4,0])

    ax1.set_xlim([rec_start, rec_stop])
    ax2.set_xlim([rec_start, rec_stop])

    ax1.set_ylabel('Neuron ID')

    ax2.set_ylabel('Rate [Hz]')
    ax2.set_xlabel('Time [ms]')

    for i in range(n_rec_in):
        ax1.plot(spikes_in[i],
                 i*np.ones(len(spikes_in[i])),
                 linestyle='',
                 marker='o',
                 color='r',
                 markersize=1)
    for i in range(n_rec_ex):
        ax1.plot(spikes_ex[i],
                 (i + n_rec_in)*np.ones(len(spikes_ex[i])),
                 linestyle='',
                 marker='o',
                 color='b',
                 markersize=1)

    ax2 = ax2.hist(spikes_ex_total,
                   range=(rec_start,rec_stop),
                   bins=int(rec_stop - rec_start))

    plt.tight_layout(pad=1)

    plt.savefig('raster.png')

def rate(spikes_ex, spikes_in, rec_start, rec_stop):
    spikes_ex_total = list(itertools.chain(*spikes_ex))
    spikes_in_total = list(itertools.chain(*spikes_in))
    spikes_total = spikes_ex_total + spikes_in_total

    n_rec_ex = len(spikes_ex)
    n_rec_in = len(spikes_in)

    time_diff = (rec_stop - rec_start)/1000.
    average_firing_rate = (len(spikes_total)
                           /(n_rec_ex + n_rec_in))
    print(f'Average firing rate: {average_firing_rate} Hz')

