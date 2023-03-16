## Import libraries
from matplotlib import pyplot as plt
import numpy as np
import nest
import itertools
import random

## 
# Create classes
class Network:
    def __init__(self, resolution, rec_start, rec_stop):
        self.__populations = []
        self.__multimeter = []
        self.__spike_recorder = []
        self.__labels = []
        self.__nrec = []
        self.rec_start = rec_start
        self.rec_stop = rec_stop
        self.resolution = resolution

    def addpop(self, neuron_type, num_neurons, neuron_params, label, 
               record_from_pop=True, nrec=0.):
        """
        Adds a population to the network with given parameters

        Parameters
        ----------
        neuron_type : string
            The neuron model name
        num_neurons : int
            number of neurons in the specified population
        neuron_params : dictionary or list of dictionaries
            parameters for the population. Randomly sampled for each neuron if list.
        label : string
            "E" for excitatory population, "I" for inhibitory population.
        record_from_pop : boolean, optional
            Whether or not to record from this population. The default is True.
        nrec : int, optional
            number of neurons to record from. The default is 0.

        Returns
        -------
        None.

        """
        
        ## If we get a list with parameter dictionaries sample randomly from them until we have
        ## a list of the size of num_neurons. Nest only accepts one dictionary for all
        ## or a list of dictionaries with size equal to the size of the population.
        if not isinstance(neuron_params, dict):
            neuron_params = random.choices(neuron_params, k=num_neurons)
        
        self.__nrec.append(nrec)
        
        ## Create the neuronal population
        newpop = nest.Create(neuron_type, num_neurons, neuron_params)
        ## If record_from_pop is true, also connect the spike recorder and multimeter to it
        if record_from_pop:
            ## Create recording devices
            sr = nest.Create("spike_recorder")
            mm = nest.Create("multimeter",
                             params={"interval": self.resolution,
                             "record_from": ["I_syn"]})
            
            ## Set start/stop times
            sr.start = self.rec_start
            sr.stop = self.rec_stop
            mm.start = self.rec_start
            mm.stop = self.rec_stop
            
            ## Connect devices to the new population
            nest.Connect(newpop[:nrec], sr)
            nest.Connect(mm, newpop[:nrec])
            
            ## append to list for data extraction later
            self.__multimeter.append(mm)
            self.__spike_recorder.append(sr)
        ## Add it to internal list of populations
        self.__populations.append(newpop)
        self.__labels.append(label)
    
    def connect(self, popone, poptwo, conn_specs, syn_specs):
        """
        Connects one population to another with given specifications.

        Parameters
        ----------
        popone : lib.hl_api_types.NodeCollection
            source nest node collection
        poptwo : lib.hl_api_types.NodeCollection
            target nest node collection
        conn_specs : dictionary
            dictionary containing connection specs
        syn_specs : dictionary
            dictionary containing synaptic specs

        Returns
        -------
        None.

        """
        nest.Connect(popone, poptwo, conn_spec=conn_specs, syn_spec=syn_specs)
    
    def connect_all(self, conn_specs, synapse_type, syn_specs):
        """
        Connect a vector containing populations with each other with given
        connectivity and synaptic specifications.

        Parameters
        ----------
        conn_specs : numpy ndarray
            connectivity matrix
        syn_specs : numpy ndarray
            specifications for each synapse

        Returns
        -------
        None.

        """
        r = list(range(len(self.__populations)))
        R = itertools.product(r,r)
        
        for x,y in R:
            #print(f"Connecting population {x} to population {y} with a connection probability of {conn_specs[x,y]} with synapse type {syn_specs[x,y]}")
            if synapse_type[x,y] == "E":
                receptor_type = 1
            else:
                receptor_type = 2
            ## Draw weights

            w_min = 0.0
            w_max = np.Inf
            weight = nest.math.redraw(nest.random.normal(
                mean = syn_specs[x,y],
                std=max(abs(syn_specs[x,y]*0.1), 1e-10)),
                min=w_min,
                max=w_max)
            delay = nest.math.redraw(nest.random.normal(
                mean = 1.5,
                std=abs(1.5*0.1)),
                min=nest.resolution, # Why would we do this? -> - 0.5 * nest.resolution,
                max=np.Inf)
            
            
            nest.Connect(self.__populations[x],
                         self.__populations[y],
                         conn_spec = {'rule': 'fixed_indegree', 
                                      'indegree': int(conn_specs[x, y] * len(self.__populations[x]))},
                         syn_spec = {"weight": weight, "delay": delay, "receptor_type": receptor_type})
        
                

    def add_stimulation(self, source, target, weight):
        """
        Adds excitatory stimulation of the specified type to the specified target
        population

        Parameters
        ----------
        source : dictionary
            Contains the type of stimulation source and the stimulation rate.
        target : int
            Specifies which population to connect to (index in the list of populations)
        weight: float
            Specifies the synaptic weight of the connection

        Returns
        -------
        None.

        """
        stimulus = nest.Create(source['type'])
        stimulus.rate = source['rate']
        nest.Connect(stimulus, self.__populations[target], conn_spec={'rule': 'all_to_all'},  syn_spec={'receptor_type': 1,
                                                                                                'weight': weight})
    
    def get_pops(self):
        """
        returns a list of all populations in the network, in the order in which they were added.

        Returns
        -------
        TYPE: list
            A list of node collections.

        """
        return self.__populations
    
    def get_last_added_population(self):
        """
        returns the population that was last added to the network

        Returns
        -------
        TYPE: lib.hl_api_types.NodeCollection
            A nest node collection

        """
        return self.__populations[-1]
    
    def simulate(self, t_sim):
        """
        Runs the simulation for t_sim miliseconds.

        Parameters
        ----------
        t_sim : float
            Simulation time in miliseconds

        Returns
        -------
        None.

        """
        nest.Simulate(t_sim)
    
    def get_labels(self):
        """
        Returns the labels for the populations in the network

        Returns
        -------
        list
            A list containing the labels indicating either excitatory or inhibitory populations.

        """
        return self.__labels
    
    def get_nrec(self):
        """
        Returns the number of neurons recorded from each population in the order they were created.

        Returns
        -------
        __nrec : list
            List of integers

        """
        return self.__nrec

    def get_data(self):
        """
        Extracts multimeter and spikerecording data from the network and returns it in organized form.

        Returns
        -------
        mmlist : list
            Contains the 'events' dictionaries for each individual neuronal population
        data : list
            Contains a list with sorted spike times for each individual neuron

        """
        ################################################
        ## Get spike data
        spike_list = [nest.GetStatus(spk)[0]['events'] for spk in self.__spike_recorder]
        ################################################
        ## Get multimeter data
        mm_list = [nest.GetStatus(d)[0]['events'] for d in self.__multimeter]
        
        return mm_list, spike_list
    
def prep_spikes(spike_list, network):
    ## Create a list containing empty lists with size equal to nrec of that population.
    ## For each of those lists get the range of IDs from the population
    ## The IDs are then lowest ID of the population + nrec
    ## Fill in the data for the corresponding senders == IDs
    ## => 
    ## Data storage:
    #nrec = [len(a['senders']) for a in spike_list]
    data = []
    pops = network.get_pops()
    nrec = network.get_nrec()
        
    ## Divide spike times to populations
    for i in range(len(pops)):
        
        spike_times = spike_list[i]
        
        #self.IDs = np.unique(nest.GetStatus(self.__spike_recorder[i])[0]["events"]["senders"])
        
        ## Create a list containing empty lists with size equal to nrec of that population.
        ## For each of those lists get the range of IDs from the population
        ## The IDs are then lowest ID of the population + nrec
        ## Fill in the data for the corresponding senders == IDs
        ## => 
        tmp = [[]] * nrec[i]
        IDs = list(pops[i].get(['global_id']).values())[0]
        mn = min(IDs)
        mx = mn + nrec[i]
        
        # times = []
        # self.min = min(self.IDs)
        # self.max = max(self.IDs)
        
        ## Get sender IDs and times (in ms)
        senders = spike_times['senders']
        times = spike_times['times']
        
        #print(f"times: {times}")
        ## Sort to make the mask work
        order = np.argsort(senders)
        senders_sorted = np.array(senders)[order]
        times_sorted = np.array(times)[order]
        
        i = 0
        for n in range(mn, mx):
            tmp[0].append(times_sorted[senders_sorted == n])
            #print(f"min: {mn}\nmax: {mx}\nsenders: {min(senders)} to {max(senders)}")
            i += 1
        
        
        data.append(tmp[0])
        
    # import pickle
    # with open("prepspikes_pre", 'wb') as f:
    #     pickle.dump(spike_list, f)
    # with open("prepspikes_post", 'wb') as f:
    #     pickle.dump(data, f)
    return data
    
    

# Helper functions
def raster(spikes, rec_start, rec_stop, colors, nrec, label, figsize=(9, 5)):
    """
    Draws the scatterplot for the spiketimes of each neuronal population as well as
    a histogram of spiketimes over all neurons.

    Parameters
    ----------
    spikes : nested list
        Should be a list containing a separate list with spike times for each neuron
    rec_start : float
        starting time of the recording in ms.
    rec_stop : float
        stopping time of the recording in ms.
    colors : list
        contains the list of colors for each population
    figsize : TYPE, optional
        DESCRIPTION. The default is (9, 5).

    Returns
    -------
    None.

    """
    #spikes_total = list(itertools.chain(*spikes))
    
    # An array containing all the arrays for each neuron
    spikes_total = [element for sublist in spikes for element in sublist]
    nrec_lst = []

    ## Get the size of each population
    for i in spikes:
        nrec_lst.append(len(i))
        
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(5, 1)

    ax1 = fig.add_subplot(gs[:4,0])
    ax2 = fig.add_subplot(gs[4,0])

    ax1.set_xlim([rec_start, rec_stop])
    ax2.set_xlim([rec_start, rec_stop])
    
    ax1.set_ylim([0, sum(nrec_lst)])
    ax1.set_ylabel('Neuron ID')
    ax1.invert_yaxis()
    
    ax2.set_ylabel('Rate [Hz]')
    ax2.set_xlabel('Time [ms]')
    
    for j in range(len(nrec_lst)): ## for each population
        for i in range(nrec_lst[j]): ## Get the size of the population
            ax1.plot(spikes_total[(i+ sum(nrec_lst[:j]))],
                (i + sum(nrec_lst[:j]))*np.ones(len(spikes_total[i+ sum(nrec_lst[:j])])),
                linestyle='',
                marker='o',
                color=colors[j],
                markersize=1)
    
    spikes_hist = list(itertools.chain(*[element for sublist in spikes for element in sublist]))
    ax2 = ax2.hist(spikes_hist,
                   range=(rec_start,rec_stop),
                   bins=int(rec_stop - rec_start))
    
    ## Add y-tick labels indicating layers
    nrec = np.cumsum(nrec)
    ticks = dict()
    layer = ["Layer 1", "Layer 2/3", "Layer 4", "Layer 5", "Layer 6"]
    l = 0
    for e in range(len(nrec)):
        if e == 0:
            ticks[nrec[e]] = layer[l]
            l += 1
        elif label[e] == "E":
            ## Display the label in the middle of the excitatory layers instead of
            ## The beginning or end
            ticks[int((nrec[e]+nrec[e-1])/2)] = layer[l]
            l += 1
            

    ax1.set_yticks(list(ticks.keys()))
    labels = [ticks[t] for i,t in enumerate(list(ticks.keys()))]
    ## or 
    # labels = [dic.get(t, ticks[i]) for i,t in enumerate(ticks)]
    
    ax1.set_yticklabels(labels)
    
    plt.tight_layout(pad=1)

    plt.savefig('simresults/raster.png')
  
def rate(spikes, rec_start, rec_stop):
    """
    Returns the average rate of spiking for the network in Hz.
    
    Parameters
    ----------
    spikes : nested list
        Should be a list containing a separate list with spike times for each neuron
    rec_start : float
        starting time of the recording in ms.
    rec_stop : float
        stopping time of the recording in ms.

    Returns
    -------
    fr : float.
        The mean firing rate of the network during the recorded period in Hz.

    """
    
    spikes_total = list(itertools.chain(*[element for sublist in spikes for element in sublist]))
    
    nrec_total = 0
    for i in spikes:
        nrec_total += len(i)

    time_diff = (rec_stop - rec_start)/1000.
    fr = (len(spikes_total)
          /time_diff
          /(nrec_total))
    print(f'Average firing rate: {fr} Hz')
    return fr

def approximate_lfp_timecourse(data, times):
    """
    This function calculates the approximated lfp timecourse for the current layer using the
    methodology from Mazzoni et al. (2015). Amplitude depends on the depth and location
    of the recording electrode and should be estimated separately to save processing resources.
    https://doi.org/10.1371%2Fjournal.pcbi.1004584
    
    Parameters
    ----------
    data : list
        Should contain the 'events' dictionaries for each neuronal population
        of the current layer and a label entry indicating excitatory ("E") or otherwise
    label: list
        Contains the label of each neuronal population in data indicating
        whether they are excitatory or inhibitory

    Returns
    -------
    normalized : numpy ndarray
        A numpy ndarray containing the approximated normalized lfp timecourse
        for the current layer

    """
    
    
    ## We are using a delay of 6ms in the excitatory populations to calculate the LFP
    ## (See the Mazzoni paper)
    ## Get the positions for the data used in the excitatory populations. Dependent on resolution
    
    delay = np.argwhere(times - min(times) >= 6)
    delay = delay.reshape(delay.shape[0],)    
    
    ## initialize arrays that will contain the sums of excitatory and inhibitory currents
    cin = np.zeros((len(delay),))
    cex = np.zeros((len(delay),))

    
    ## Go through all different neuronal populations of the current layer
    for d in range(len(data)):
        ## get the relevant data for the population
        senders = np.array(data[d]["senders"])
        I_syn = np.array(data[d]["I_syn"])
        l = data[d]["label"]
        
        ## neuronal populations always have sequential IDs upon creation
        ## so we just need the first and last ID of the current population
        mn = min(senders)
        mx = max(senders)
        
        ## get a list containing the synaptic current
        ## recordings per neuron
        ## Heaviest bottleneck right now: I'll get you a coffee if you have better ideas
        mmdata = [I_syn[senders == s] for s in list(range(mn, mx+1))]
        
        ## The currents are already sorted by time, just
        ## sum them up at each point in time and save them in an array
        csum = np.zeros(len(delay))
        if l == "E":
            csum = np.array([sum(mmdata)[delay]])
        else:
            csum = np.array([sum(mmdata)[delay - min(delay)]])
        csum = csum.reshape(csum.shape[1],)
            
        ## Append to list. It now contains the sums of currents from each
        ## individual neuronal population
        if l == "E":
            cex += csum
        else:
            cin += csum
    
    # Apply the formula: norm[sum(current_ex) - 1.65 * sum[current_inh]]
    normalized = normalize(cex - 1.65*cin)
    
    ## return the normalized LFP timecourse.
    return normalized

def normalize(data):
    """
    Mean subtracted normalization procedure for a given array

    Parameters
    ----------
    data : list or ndarray
        array to be converted

    Returns
    -------
    normalized : numpy ndarray
        converted array

    """
    mean = np.mean(data)
    ms = np.subtract(data,mean)
    ## https://stackoverflow.com/questions/1735025/how-to-normalize-a-numpy-array-to-within-a-certain-range
    normalized = (ms - np.min(ms))/np.ptp(ms)
    
    return normalized

def get_isi(spike_times):
    """
    Calculates the interspike interval (isi) for all populations

    Parameters
    ----------
    spike_times : list
        A list containing the arrays of spike times for each population

    Returns
    -------
    list
        Returns a list of interspike intervalls

    """
    return [np.diff(x) if len(x) > 1 else [] for x in spike_times]

def get_synchrony(population, start, stop):
    spike_counts = np.histogram(np.concatenate(population), bins=np.arange(start, stop+3, 3))[0]
    mean_spike_count = np.mean(spike_counts)
    if mean_spike_count == 0:
        ## Penalize no firing rate more than high synchrony
        return 10000
    var_spike_count = np.var(spike_counts)
    vm_ratio = var_spike_count / mean_spike_count
    return vm_ratio


def get_irregularity(population):
    """
    Calculates an irregularity measure based on Potjans & Diesmann (2014)
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3920768/

    Parameters
    ----------
    population : list
        A list containing the arrays of spike times for each neuron in the population

    Returns
    -------
    cv : float
        A measure of irregularity, bounded between 0 and 1, where 1 is the highest and 0 the lowest.
        Returns -1 if there is no data for that population

    """
    isi = [np.diff(np.sort(neuron)) for neuron in population]
    cv  = [np.std(i) / np.mean(i) for i in isi if not np.isnan(np.std(i) / np.mean(i))]
    # isi = list(itertools.chain(*population))
    # isi = np.concatenate([np.diff(neuron) for neuron in population if len(neuron) > 1])
    if len(cv) == 0:
        ## Penalize no firing rate harder than no irregularity
        return 10000
    
    irregularity = np.mean(cv)

    return irregularity

def get_firing_rate(spike_times, start, stop):
    spikes_total = list(itertools.chain(*spike_times))
    dt = (stop - start) / 1000
    afr = len(spikes_total)/dt/len(spike_times)
    
    ## Penalize silent populations
    if afr == 0:
        return 10000
    
    return afr

def join_results(simres):
    """
    Gathers and joins the results from the distributed simulation to be used in the rest of the script.

    Parameters
    ----------
    simres : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # Determine the number of inner lists and dictionaries in each inner list
    merged_dicts = [{k: [] for k in simres[0][0].keys()} for i in range(17)]

    # Merge the dictionaries
    for inner_list in simres:
        for i, d in enumerate(inner_list):
            for k, v in d.items():
                merged_dicts[i][k].extend(v)
    # import pickle
    # with open("join_results", 'wb') as f:
    #     pickle.dump(simres, f)
    return merged_dicts
    
