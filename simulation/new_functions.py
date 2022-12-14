# Import libraries
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nest
import itertools
import random

## 

# Create classes
class Network():
    def __init__(self, ext_rate):
        self.__populations = []
        self.__devices = []
        self.ext_rate = ext_rate
        self.spike_recorder = nest.Create("spike_recorder")

    def addpop(self, neuron_type, num_neurons, neuron_params, record_from_pop=False, nrec=0.):
        ## Create the neuronal population
        newpop = nest.Create(neuron_type, num_neurons, neuron_params)
        ## If record_from_pop is true, also connect the spike recorder to it
        if record_from_pop:
            nest.Connect(newpop[:nrec], self.spike_recorder)
        ## Add it to internal list of populations
        self.__populations.append(newpop)
    
    def connect(self, popone, poptwo, conn_specs, syn_specs):
        nest.Connect(popone, poptwo, conn_spec=conn_specs, syn_spec=syn_specs)
    
    def connect_all(self, populations, conn_specs, syn_specs='default'):
        ## Connect a vector containing populations with other populations
        ## with given parameters.
        ## Accepts either one synapse model for all connections or a matrix
        ## of synapse models for each individual connection
        if isinstance(syn_specs, str):
           for x in range(len(populations)):
               for y in range(len(populations)):
                   nest.Connect(populations[x], 
                                populations[y],
                                conn_spec = {'rule': 'fixed_indegree', 'indegree': int(conn_specs[y,x] * (len(populations[y]) + len(populations[x])))})
        elif isinstance(syn_specs, dict):
            for x in range(len(populations)):
                for y in range(len(populations)):
                    nest.Connect(populations[x], 
                                 populations[y],
                                 conn_spec = {'rule': 'fixed_indegree', 'indegree': int(conn_specs[y,x] * (len(populations[y]) + len(populations[x])))},
                                 syn_spec = syn_specs)
        else:
            for x in range(len(populations)):
                for y in range(len(populations)):
                    nest.Connect(populations[x], 
                                 populations[y],
                                 conn_spec = {'rule': 'fixed_indegree', 'indegree': int(conn_specs[y,x] * (len(populations[y]) + len(populations[x])))},
                                 syn_spec = syn_specs[y,x])

    def create(self):
        self.stimulus = nest.Create("poisson_generator")
        self.stimulus.rate = self.ext_rate
        nest.Connect(self.stimulus, self.__populations[0],
                     #conn_spec={'rule': 'fixed_indegree', 'indegree': self.c_ex},
                     syn_spec={'receptor_type': 1,
                               'weight': 1.})
    
    def get_pops(self):
        return self.__populations
    
    def get_last_added_population(self):
        return self.__populations[-1]
    
    def simulate(self, t_sim):
        # Simulate for t_sim miliseconds
        nest.Simulate(t_sim)

    def get_data(self):
        ## Get the data f rom the spike recorder
        self.spike_times = nest.GetStatus(self.spike_recorder)
        
        ## Data storage:
        self.data = []
        
        ## Divide spike times to populations
        for i in range(len(self.__populations)):
            
            self.IDs = list(self.__populations[i].get(['global_id']).values())[0]
            self.min = min(self.IDs)
            self.max = max(self.IDs)
            
            ## Get sender IDs and times (in ms)
            self.senders = self.spike_times[0]['events']['senders']
            self.times = self.spike_times[0]['events']['times']
            
            ## Select only the relevant ones for this population
            self.current_senders = self.senders[(self.senders >= self.min) & (self.senders <= self.max)]
            
            ## get sorted index array for senders
            self.idx = np.argsort(self.current_senders)
            ## sort sender array
            self.senders_sorted = self.current_senders[self.idx]
            ## sort spike times
            self.times_sorted = self.times[self.idx]

            ## Create an array with each neurons spike times for each row
            self.spike_times_sorted = np.split(self.times_sorted, np.where(np.diff(self.senders_sorted) > 0)[0] + 1)
            
            ## Store to data
            self.data.append(self.spike_times_sorted)
            
        return self.data

# Helper functions
def raster(spikes, rec_start, rec_stop, figsize=(9, 5)):

    #spikes_total = list(itertools.chain(*spikes))
    spikes_total = [element for sublist in spikes for element in sublist]
    nrec_lst = []
    
    for i in spikes:
        nrec_lst.append(len(i))
        
    
    
    #n_rec_ex = len(spikes_ex)
    #n_rec_in = len(spikes_in)

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 1)

    ax1 = fig.add_subplot(gs[:4,0])
    ax2 = fig.add_subplot(gs[4,0])

    ax1.set_xlim([rec_start, rec_stop])
    ax2.set_xlim([rec_start, rec_stop])

    ax1.set_ylabel('Neuron ID')

    ax2.set_ylabel('Rate [Hz]')
    ax2.set_xlabel('Time [ms]')
    
    color_list = []
    for i in range(len(nrec_lst)):
      r = random.randint(0,255)/255
      g = random.randint(0,255)/255
      b = random.randint(0,255)/255
      color_list.append([r,g,b])
    
    for j in range(len(nrec_lst)):
        for i in range(nrec_lst[j]):
            ax1.plot(spikes_total[i],
                (i + sum(nrec_lst[:j]))*np.ones(len(spikes_total[i])),
                linestyle='',
                marker='o',
                color=color_list[j],
                markersize=1)

    spikes_total = list(itertools.chain(*[element for sublist in spikes for element in sublist]))
    ax2 = ax2.hist(spikes_total,
                   range=(rec_start,rec_stop),
                   bins=int(rec_stop - rec_start))

    plt.tight_layout(pad=1)

    plt.savefig('raster.png')

def rate(spikes, rec_start, rec_stop):
    spikes_total = list(itertools.chain(*[element for sublist in spikes for element in sublist]))

    
    nrec_total = 0
    for i in spikes:
        nrec_total += len(i)

    time_diff = (rec_stop - rec_start)/1000.
    average_firing_rate = (len(spikes_total)
                           /(nrec_total))
    print(f'Average firing rate: {average_firing_rate} Hz')