# Import libraries
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import nest
import itertools
import random

## 
# Create classes
class Network:
    def __init__(self):
        self.__populations = []
        self.__devices = []
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
    
    def connect_all(self, conn_specs, syn_specs):
        ## Connect a vector containing populations with other populations
        ## with given parameters.
        ## Accepts either one synapse model for all connections or a matrix
        ## of synapse models for each individual connection
        r = list(range(len(self.__populations)))
        R = itertools.product(r,r)
        for x,y in R:
            nest.Connect(self.__populations[x],
                         self.__populations[y],
                         conn_spec = {'rule': 'fixed_indegree', 
                                      'indegree': int(conn_specs[y, x] * len(self.__populations[x]))},
                         syn_spec = syn_specs[y, x])
        
                

    def add_stimulation(self, source, target):
        stimulus = nest.Create(source['type'])
        stimulus.rate = source['rate']
        nest.Connect(stimulus, self.__populations[target], conn_spec={'rule': 'all_to_all'},  syn_spec={'receptor_type': 1,
                                                                                                'weight': 0.1})
    
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
    
    # An array containing all the arrays for each neuron
    spikes_total = [element for sublist in spikes for element in sublist]
    nrec_lst = []
    
    ## Get the size of each population
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
    
    color_list = ['b', 'r']
    # for i in range(len(nrec_lst)):
    #   r = random.randint(0,255)/255
    #   g = random.randint(0,255)/255
    #   b = random.randint(0,255)/255
    #   color_list.append([r,g,b])
    
    
    print(nrec_lst)
    for j in range(len(nrec_lst)): ## for each population
        for i in range(nrec_lst[j]): ## Get the size of the population
            ax1.plot(spikes_total[(i+ sum(nrec_lst[:j]))],
                (i + sum(nrec_lst[:j]))*np.ones(len(spikes_total[i+ sum(nrec_lst[:j])])),
                linestyle='',
                marker='o',
                color=color_list[j],
                markersize=1)

    spikes_hist = list(itertools.chain(*[element for sublist in spikes for element in sublist]))
    ax2 = ax2.hist(spikes_hist,
                   range=(rec_start,rec_stop),
                   bins=int(rec_stop - rec_start))

    plt.tight_layout(pad=1)

    plt.savefig('raster.png')

def rate(spikes, rec_start, rec_stop):
    spikes_total = list(itertools.chain(*[element for sublist in spikes for element in sublist]))
    print(len(spikes_total))
    
    nrec_total = 0
    for i in spikes:
        nrec_total += len(i)
    print(nrec_total)

    time_diff = (rec_stop - rec_start)/1000.
    average_firing_rate = (len(spikes_total)
                           #/time_diff
                           /(nrec_total))
    print(f'Average firing rate: {average_firing_rate} Hz')