## Import libraries
from matplotlib import pyplot as plt
import numpy as np
import nest
import itertools

## 
# Create classes
class Network:
    def __init__(self, resolution, rec_start, rec_stop):
        self.__populations = []
        self.__devices = []
        self.rec_start = rec_start
        self.rec_stop = rec_stop
        self.resolution = resolution
        self.spike_recorder = nest.Create("spike_recorder")
        self.spike_recorder.start = self.rec_start
        self.spike_recorder.stop = self.rec_stop

    def addpop(self, neuron_type, num_neurons, neuron_params, neuron_positions, 
               record_from_pop=True, nrec=0.):
        """
        Adds a population to the network with given parameters

        Parameters
        ----------
        neuron_type : string
            The neuron model name
        num_neurons : int
            number of neurons in the specified population
        neuron_params : dictionary
            parameters for the population
        neuron_positions : nest spatial data
            nest spatial data
        record_from_pop : boolean, optional
            Whether or not to record from this population. The default is True.
        nrec : int, optional
            number of neurons to record from. The default is 0..

        Returns
        -------
        None.

        """
        
        ## Create the neuronal population
        newpop = nest.Create(neuron_type, num_neurons, neuron_params, positions=neuron_positions)
        ## If record_from_pop is true, also connect the spike recorder and multimeter to it
        if record_from_pop:
            nest.Connect(newpop[:nrec], self.spike_recorder)
            mm = nest.Create("multimeter",
                             params={"interval": self.resolution,
                             "record_from": ["V_m", "I_syn"]})
            mm.start = self.rec_start
            mm.stop = self.rec_stop
            nest.Connect(mm, newpop)
            self.__devices.append(mm)
        ## Add it to internal list of populations
        self.__populations.append(newpop)
    
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
    
    def connect_all(self, conn_specs, syn_specs):
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
            nest.Connect(self.__populations[x],
                         self.__populations[y],
                         conn_spec = {'rule': 'fixed_indegree', 
                                      'indegree': int(conn_specs[y, x] * len(self.__populations[x]))},
                         syn_spec = syn_specs[y, x])
        
                

    def add_stimulation(self, source, target):
        """
        Adds excitatory stimulation of the specified type to the specified target
        population

        Parameters
        ----------
        source : dictionary
            Contains the type of stimulation source and the stimulation rate.
        target : int
            Specifies which population to connect to (index in the list of populations)

        Returns
        -------
        None.

        """
        stimulus = nest.Create(source['type'])
        stimulus.rate = source['rate']
        nest.Connect(stimulus, self.__populations[target], conn_spec={'rule': 'all_to_all'},  syn_spec={'receptor_type': 1,
                                                                                                'weight': 1.})
    
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
            
            times = []
            for n in range(self.min, self.max+1):
                times.append(self.times[self.senders == n])
            
            self.data.append(times)
            
            ################################################
            ## Get multimeter data
            mmlist = []
            for d in self.__devices:
                mmlist.append(nest.GetStatus(d)[0]["events"])
            
        return mmlist, self.data

# Helper functions
def raster(spikes, rec_start, rec_stop, figsize=(9, 5)):
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

    ax1.set_ylabel('Neuron ID')

    ax2.set_ylabel('Rate [Hz]')
    ax2.set_xlabel('Time [ms]')
    
    color_list = ['b', 'r']
    # for i in range(len(nrec_lst)):
    #   r = random.randint(0,255)/255
    #   g = random.randint(0,255)/255
    #   b = random.randint(0,255)/255
    #   color_list.append([r,g,b])
    
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
    """
    Displays the average rate of spiking for the network in Hz.
    
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
    None.

    """
    
    spikes_total = list(itertools.chain(*[element for sublist in spikes for element in sublist]))
    
    nrec_total = 0
    for i in spikes:
        nrec_total += len(i)
    print(nrec_total)

    time_diff = (rec_stop - rec_start)/1000.
    average_firing_rate = (len(spikes_total)
                           /time_diff
                           /(nrec_total))
    print(f'Average firing rate: {average_firing_rate} Hz')
    

def approximate_lfp_timecourse(data):
    """
    This function calculates the approximated lfp timecourse for the current layer using the
    methodology from Mazzoni et al. (2015). Amplitude depends on the depth and location
    of the recording electrode and should be estimated separately to save processing resources.
    https://doi.org/10.1371%2Fjournal.pcbi.1004584
    
    Parameters
    ----------
    data : list
        Should contain the 'events' dictionaries for each neuronal population
        of the current layer

    Returns
    -------
    normalized : numpy ndarray
        A numpy ndarray containing the approximated normalized lfp timecourse
        for the current layer

    """
    
    ## Go through all different neuronal populations of the current layer
    sums = []
    mmall = []
    for d in data:
        ## get the relevant data for the population
        senders = np.array(d["senders"])
        I_syn = np.array(d["I_syn"])
        
        ## neuronal populations always have sequential IDs upon creation
        ## so we just need the first and last ID of the current population
        mn = min(senders)
        mx = max(senders)
        
        ## get a list containing the synaptic current
        ## recordings per neuron
        mmdata = []
        i = 0
        for s in range(mn, mx+1):
            mmdata.append( I_syn[senders == s] )
            i += 1
        
        ## The currents are already sorted by time, just
        ## sum them up at each point in time and save them in an array
        currentsum = np.zeros(len(mmdata[0]))
        for i in range(len(mmdata[0])):
            tmpsum = 0
            for n in mmdata:
                tmpsum += n[i]
            currentsum[i] = tmpsum
        
        ## Append to list. It now contains the sums of currents from each
        ## individual neuronal population
        sums.append(currentsum)
        mmall.append(mmdata)
    
    
    ## Apply the formula: norm[sum(current_ex) - 1.65 * sum[current_inh]]
    subtracted = np.subtract(sums[0], 1.65*sums[1])
    
    normalized = normalize(subtracted)
    for x in mmall:
        for i in range(len(x)):
            x[i] = normalize(x[i])
    
    ## return the normalized LFP timecourse.
    return normalized, mmall

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
    normalized = 2.*(ms - np.min(ms))/np.ptp(ms)-1
    
    return normalized
