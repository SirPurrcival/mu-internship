from matplotlib import pyplot as plt
import numpy as np
import nest
import itertools
import random

## 
# Create classes
class Network:
    def __init__(self, p):
        self.populations = {}
        self.thalamic_population = {}
        self.nrec = []
        self.rec_start = p['rec_start']
        self.rec_stop = p['rec_stop']
        self.resolution = p['resolution']
        self.opt = p['opt_run']
        self.g = p['g']
        self.record_to = p['record_to']
        
        #######################################################################
        ## Create recording devices
        ## Spike recorder
        self.sr = nest.Create("spike_recorder",
                    params = {'record_to': self.record_to,
                              'start'    : self.rec_start,
                              'stop'     : self.rec_stop,
                              'label'    : "spike_recorder"
                              })
        ## Multimeter
        self.mm = nest.Create("multimeter",
                    params={'interval'   : self.resolution,
                            'record_to'  : self.record_to,
                            'start'      : self.rec_start,
                            'stop'       : self.rec_stop,
                            'label'      : 'multimeter',
                            'record_from': ["V_m"]
                            })

    def addpop(self, neuron_type, pop_name ,num_neurons, neuron_params=None, 
               record=False, nrec=0.):
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
        if isinstance(neuron_params, list):
            neuron_params = random.choices(neuron_params, k=num_neurons)
        
        
        ## Create the neuronal population
        newpop = nest.Create(neuron_type, num_neurons, neuron_params)
        
        if not neuron_params == None:
            self.nrec.append(nrec)
            
            ## Change around V_m as done in Potjans & Diesmann
            nest.SetStatus(newpop, 'V_m', np.random.normal(-58., 10.0, num_neurons))
        
            
            ## If record_from_pop is true, also connect the spike recorder and multimeter to it
            if record:
                ## Connect devices to the new population
                nest.Connect(newpop[:nrec], self.sr)
                
                ## Connect devices to the new population
                nest.Connect(self.mm, newpop[:nrec])
                
            ## Add it to internal list of populations
            self.populations[pop_name] = newpop
            
        else:
            self.thalamic_population[pop_name] = newpop
    
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
    
    def connect_all(self, names, conn_specs, synapse_type, syn_specs):
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
        r = list(range(len(self.populations)))
        R = itertools.product(r, r)
        
        for source,target in R:
            #print(f"Connecting population {x} to population {y} with a connection probability of {conn_specs[x,y]} with synapse type {syn_specs[x,y]}")
            if synapse_type[source,target] == "E":
                receptor_type = 1
                w_min = 0.0
                w_max = np.inf
            else:
                receptor_type = 2
                w_min = np.NINF
                w_max = 0.0
                
            weight = nest.math.redraw(nest.random.normal(
                mean = syn_specs[source,target],
                std=max(abs(syn_specs[source,target]*0.1), 1e-10)),
                min=w_min,
                max=w_max)
            delay = nest.math.redraw(nest.random.normal(
                mean = 1.5 if synapse_type[source,target] == "E" else 0.75,
                std=abs(1.5*0.5) if synapse_type[source,target] == "E" else abs(0.75*0.5)),
                min=self.resolution, 
                max=np.Inf)

            nest.Connect(self.populations[names[source]],
                         self.populations[names[target]],
                         conn_spec = {'rule': 'fixed_indegree', 
                                      'indegree': int(conn_specs[source, target] * len(self.populations[names[source]]))},
                         syn_spec = {"weight": weight, "delay": delay})#, "receptor_type": receptor_type})
        
                

    def add_stimulation(self, source, target, c_specs, s_specs):
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
        stimulus.start = source['start']
        stimulus.stop = source['stop']
        
        ##TODO: Please make this better :( 
        try:
            target = self.populations[target]
        except:
            target = self.thalamic_population[target]
        
        nest.Connect(stimulus, target,
                     conn_spec = c_specs,
                     syn_spec = s_specs)
        
    def add_dc_stimulation(self, source, target):
        stimulus = nest.Create(source['type'])
        stimulus.amplitude = source['amplitude']
        nest.Connect(stimulus, self.populations[target], conn_spec={'rule': 'all_to_all'})
    
    def get_pops(self):
        """
        returns a list of all populations in the network, in the order in which they were added.

        Returns
        -------
        TYPE: list
            A list of node collections.

        """
        return self.populations
    
    def get_last_added_population(self):
        """
        returns the population that was last added to the network

        Returns
        -------
        TYPE: lib.hl_api_types.NodeCollection
            A nest node collection

        """
        return self.populations[-1]
    
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
    
    def get_nrec(self):
        """
        Returns the number of neurons recorded from each population in the order they were created.

        Returns
        -------
        nrec : list
            List of integers

        """
        return self.nrec

    def get_data(self):
        """
        Extracts multimeter and spikerecording data from the network 
        and returns it in organized form.

        """
        #######################################################################
        ## Fetch from files. 
        if self.record_to == 'ascii':
            mm_files = self.mm.filenames
            sr_files = self.sr.filenames
            
            sr_data = np.hstack([np.loadtxt(f, skiprows=3).T for f in sr_files])
            mm_data = np.hstack([np.loadtxt(f, skiprows=3).T for f in mm_files])
            
            sr_header = np.loadtxt(sr_files[0], dtype=str, skiprows=2, max_rows=1)
            mm_header = np.loadtxt(mm_files[0], dtype=str, skiprows=2, max_rows=1)
            
            sr_dict = {sr_header[i]:sr_data[i] for i in range(len(sr_header))}
            mm_dict = {mm_header[i]:mm_data[i] for i in range(len(mm_header))}
        #######################################################################
        ## Fetch from memory. Best used for small networks
        elif self.record_to == 'memory':
            ## Get data from the recorder
            sr_dict = nest.GetStatus(self.sr)[0]['events']
            mm_dict = nest.GetStatus(self.mm)[0]['events']
            
            ## make sure the keys are the same as if recording to ascii
            sr_dict['time_ms'] = sr_dict.pop('times')
            mm_dict['time_ms'] = mm_dict.pop('times')
            
            sr_dict['sender'] = sr_dict.pop('senders')
            mm_dict['sender'] = mm_dict.pop('senders')
        else:
            ## No other recording methods used at this point
            print("Unsupported recording type")
        
        
        return mm_dict, sr_dict
    
    def prep_spikes(self, spikes):
        ## Get data for senders and times
        times  = spikes['time_ms']
        sender = spikes['sender']
        
        ## Order by sender
        order = np.argsort(sender)
        
        times  = times[order]
        sender = sender[order]
        
        ## we want to see silent neurons
        unique_senders = np.unique(sender)
        min_sender = min(unique_senders)
        ## Add an empty list for every neuron that is silent.
        ## This requires all recorded neurons to be added sequentially
        data = [times[np.where(sender == i+min_sender)] if i + min_sender in unique_senders else np.array([]) for i in range(sum(self.nrec))]
        ## Split when the sending neuron changes
        # data = np.split(times, np.where(np.diff(sender))[0])
        
        return data
    
    def split_mmdata(self, mmdata):
        sender = np.array(mmdata['sender'])
        dict_1 = {key: [] for key in mmdata}
        dict_2 = {key: [] for key in mmdata}
        
        ## split by values
        min_sender = min(np.unique(sender))
        
        mask_1 = np.where((sender >= min_sender) & (sender <= min_sender+sum(self.nrec[:2])))
        mask_2 = np.where((sender > min_sender+sum(self.nrec[:2])) & (sender <= min_sender+sum(self.nrec[:4])))
        
        for key in mmdata:
            dict_1[key] = np.array(mmdata[key])[mask_1]
            dict_2[key] = np.array(mmdata[key])[mask_2]
        
        return dict_1, dict_2
    

# Helper functions
def raster(spikes, vm_data, rec_start, rec_stop, colors, nrec, prefix="", suffix="", figsize=(10, 7)):
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
    
    print("Start graph")
    
    # An array containing all the arrays for each neuron
    spikes_total = [element for sublist in spikes for element in sublist]
    nrec_lst = []

    ## Get the size of each population
    for i in spikes:
        nrec_lst.append(len(i))
        
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(4, 2)

    ax1 = fig.add_subplot(gs[:3,0:])
    ax2 = fig.add_subplot(gs[3,0])
    ax3 = fig.add_subplot(gs[3,1:])

    ax1.set_xlim([rec_start, rec_stop])
    ax2.set_xlim([rec_start, rec_stop])
    ax3.set_xlim([rec_start, rec_stop])
    
    ax1.set_ylim([0, sum(nrec_lst)])
    ax1.set_ylabel('Neuron ID')
    ax1.invert_yaxis()
    
    ax2.set_ylabel('Rate [Hz]')
    ax2.set_xlabel('Time [ms]')
    
    ax3.set_ylabel('V_m')
    ax3.set_xlabel('Time [ms]')

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
    
    x_values = np.arange(rec_start + 0.1, rec_stop, 0.1)
    ax3 = ax3.plot(x_values, vm_data)
    
    plt.tight_layout(pad=1)

    plt.savefig(f"simresults/{prefix}raster_{suffix}.png")
    
    plt.show()
    
def create_spectrogram(data, fs, t_start, t_end, f_min, f_max):
    _, _, _, im = plt.specgram(data, Fs=fs, NFFT=4096, noverlap=4096//2, xextent=[t_start, t_end])
    plt.colorbar(label='Power Spectral Density (dB/Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.ylim(f_min, f_max)
    plt.savefig("simresults/spectrogram.png")
    plt.show()

    return im.get_array()

def spectral_density(timesteps, resolution, data, cutoff=1e7, plot=True, get_peak=False):
        ## Compute the power spectral density (PSD)
        frequencies = np.fft.fftfreq(timesteps, resolution)

        ## Compute the Fourier transform
        if plot:
            plt.figure()
            
        psd_return = []
        if type(data) == list:
            for i in range(len(data)):
                data_zero = data[i] - np.mean(data[i])
                xf = np.fft.fft(data_zero)
                psd = np.abs(xf)**2
                psd_return.append(psd)
                if plot:
                    plt.plot(frequencies, psd, label=f"Dataset {i+1}")
        else:
            data_zero = data - np.mean(data)
            xf = np.fft.fft(data_zero) 
            ## Compute the power spectral density
            psd = np.abs(xf)**2  
            ## Plot the spectral density
            if plot:
                plt.plot(frequencies, psd)
            psd_return.append(psd)
            
        if plot:
            plt.xlabel('Frequency [Hz]')
            plt.ylabel('Power Spectral Density')
            plt.xlim(0, 80)
            plt.axhline(y=cutoff, color='black', linestyle='--')
            plt.title('Spectral Density Plot')
            plt.grid(True)
            plt.show()
        
        if get_peak:
            if len(psd_return) > 1:
                ## Find indices where the greatest peak lies. If the peak is below a
                ## certain cutoff value, add -1 instead to signify no sufficient
                ## synchronous behaviour
                max_index = [np.argmax(i[:len(i)//2]) if i[np.argmax(i[:len(i)//2])] >= cutoff else -1 for i in psd_return]
                
                ## If there was no significant synchronous behaviour return 0
                ## as 'peak frequency'
                peak = np.array([frequencies[x] if x > -1 else -1 for x in max_index])
                
                ## Delete negative frequencies except -1
                peak = peak[peak>=-1]
                    
            else:
                max_index = np.argmax(psd_return[0][:len(psd_return[0])//2])
                
                if psd_return[0][max_index] >= cutoff:
                    peak = frequencies[max_index]
                    peak = peak[peak>0]
                else:
                    peak = -1
            return peak
  
def rate(spikes, rec_start, rec_stop):
    """
    Returns the average rate of spiking for the entire network in Hz.
    
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
    """
    Calculates a synchrony measure based on Potjans & Diesmann (2014)
    https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3920768/

    Parameters
    ----------
    population : list
        A list containing the arrays of spike times for each neuron in the population
    start : int
        Starting time in ms
    stop : int
        Stopping time in ms

    Returns
    -------
    vm_ratio : float
        A measure of synchrony based on Potjans & Diesmann
        Returns -1 if there is no data for that population

    """
    spike_counts = np.histogram(np.concatenate(population), bins=np.arange(start, stop+3, 3))[0]
    mean_spike_count = np.mean(spike_counts)
    if mean_spike_count == 0:
        ## Penalize no firing rate more than high synchrony
        return -1
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
        A measure of irregularity based on Potjans & Diesmann
        Returns -1 if there is no data for that population

    """
    isi = [np.diff(np.sort(neuron)) for neuron in population]
    cv  = [np.std(i) / np.mean(i) for i in isi if not np.isnan(np.std(i) / np.mean(i))]
    # isi = list(itertools.chain(*population))
    # isi = np.concatenate([np.diff(neuron) for neuron in population if len(neuron) > 1])
    if len(cv) == 0:
        ## Penalize no firing rate harder than no irregularity
        return -1
    
    irregularity = np.mean(cv)

    return irregularity

def get_firing_rate(spike_times, start, stop):
    """Calculates and returns the average firing rate of the given population and time frame.
       Sets firing rate to 10 kHz if the population is silent during that time to penalize
       the objective function."""
       
    spikes_total = list(itertools.chain(*spike_times))
    dt = (stop - start) / 1000
    afr = len(spikes_total)/dt/len(spike_times)
    
    ## Penalize silent populations
    if afr == 0:
        return 10000
    
    return afr
    
def get_spike_rate(times, params):
    bins = (np.arange(params['transient'] / params['resolution'], params['sim_time'] / params['resolution'] + 1)
            * params['resolution'] - params['resolution'] / 2)
    hist, _ = np.histogram(times, bins=bins)
    return bins, hist.astype(float)

def get_mean_spike_rate(times, params):
    times = times[times >= params['transient']]
    return times.size /  (params['sim_time'] - params['transient']) * 1000