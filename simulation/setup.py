import numpy as np
import pickle
import scipy.stats as st
from LFPy import NetworkCell
from example_network_methods import set_active_hay2011 as set_active



def setup():
    #######################################
    ## Set parameters for the simulation ##
    #######################################
    
    ## Recording and simulation parameters
    params = {
        'rec_start'  :   200.,                                                      # start point for data recording
        'rec_stop'   :   1000.,                                                      # end points for data recording
        'sim_time'   :   1000.,                                                      # Time the network is simulated in ms
        'calc_lfp'   :   False,                                                      # Flag to use LFP approximation procedure
        'verbose'    :  True,                                                       # Flag for verbose function output
        'K_scale'    :     1.,                                                      # Scaling factor for connections
        'syn_scale'  :     1.,                                                      # Scaling factor for synaptic strenghts
        'N_scale'    :     1.,                                                      # Scaling factor for the number of neurons
        'R_scale'    :     0.1,                                                     # Fraction of neurons to be recorded from
        'opt_run'    :   False,                                                     # Flag for optimizer run, run minimal settings
        'g'          :      -4.,                                                     # Excitation-Inhibition balance
        'resolution' :   2**-3,                                                     # Resolution of the simulaton
        'transient'  :     200,                                                     # Ignore the first x ms of the simulation
        'th_in'      :     0.,#902,                                                 # Thalamic input, nodes x frequency
        'num_neurons': np.array([20683, 5834, 21915, 5479, 4850, 1065, 14365, 2948]),
        'label'      : ['E', 'I', 'E', 'I', 'E', 'I', 'E', 'I'],
        'cell_params'  : {
                        'V_m'     : -58.,
                        'V_th'    : -50.,
                        'V_reset' : -65.,
                        'C_m'     : 250.,
                        't_ref'   : 2. ,
                        'tau_syn_ex' : 0.5,
                        'tau_syn_in' : 0.5,
                        'E_L':       -65.0,
                        'tau_m' :     10.0,},              # cell types, courtesy of the allen institute
        'layer_type' : [                                                             # Layer and cell types
                        'L23_E', 'L23_I',
                        'L4_E', 'L4_I',  
                        'L5_E', 'L5_I', 
                        'L6_E', 'L6_I']
        }
    
    ################################################################
    ## Specify connectivity in and between layers and populations ##
    ################################################################
    
    # Connectivity matrix                
    params['connectivity'] = np.transpose(np.array(
            [[0.1009, 0.1689, 0.0437, 0.0818, 0.0323, 0.    , 0.0076, 0.    ],
             [0.1346, 0.1371, 0.0316, 0.0515, 0.0755, 0.    , 0.0042, 0.    ],
             [0.0077, 0.0059, 0.0497, 0.135  , 0.0067, 0.0003, 0.0453, 0.    ],
             [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.    , 0.1057, 0.    ],
             [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
             [0.0548, 0.0269, 0.0257, 0.0022, 0.06  , 0.3158, 0.0086, 0.    ],
             [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364, 0.001  , 0.0034, 0.0005, 0.0277, 0.008 , 0.0658, 0.1443]]
            ))
    
    
    def get_weight(PSP_val, net_dict):
        """ Computes weight to elicit a change in the membrane potential.
        This function computes the weight which elicits a change in the membrane
        potential of size PSP_val. To implement this, the weight is calculated to
        elicit a current that is high enough to implement the desired change in the
        membrane potential.
        Parameters
        ----------
        PSP_val
            Evoked postsynaptic potential.
        net_dict
            Dictionary containing parameters of the microcircuit.
        Returns
        -------
        PSC_e
            Weight value(s).
        """
        C_m = net_dict['cell_params']['C_m']
        tau_m = net_dict['cell_params']['tau_m']
        tau_syn_ex = net_dict['cell_params']['tau_syn_ex']
    
        PSC_e_over_PSP_e = (((C_m) ** (-1) * tau_m * tau_syn_ex / (
            tau_syn_ex - tau_m) * ((tau_m / tau_syn_ex) ** (
                - tau_m / (tau_m - tau_syn_ex)) - (tau_m / tau_syn_ex) ** (
                    - tau_syn_ex / (tau_m - tau_syn_ex)))) ** (-1))
        PSC_e = (PSC_e_over_PSP_e * PSP_val)
        return PSC_e
    
    ################################
    ## Specify synapse properties ##
    ################################
    
    params['syn_type'] = np.array([["E", "E", "E", "E", "E", "E", "E", "E"],
                                   ["I", "I", "I", "I", "I", "I", "I", "I"],
                                   ["E", "E", "E", "E", "E", "E", "E", "E"],
                                   ["I", "I", "I", "I", "I", "I", "I", "I"],
                                   ["E", "E", "E", "E", "E", "E", "E", "E"],
                                   ["I", "I", "I", "I", "I", "I", "I", "I"],
                                   ["E", "E", "E", "E", "E", "E", "E", "E"],
                                   ["I", "I", "I", "I", "I", "I", "I", "I"]
                                    ])
    
    ## Synaptic strength
    params['syn_strength'] = np.array([
                                    [0.15]*8,
                                    [0.15*params['g']]*8,
                                    [0.15]*8,
                                    [0.15*params['g']]*8,
                                    [0.15]*8,
                                    [0.15*params['g']]*8,
                                    [0.15]*8,
                                    [0.15*params['g']]*8]
                                    )
    params['syn_strength'][2,0] = 0.15 * 2
    
    params['syn_strength'] = get_weight(params['syn_strength'], params)
    
    
    #######################################
    ## Background stimulation parameters ##
    #######################################

    params['ext_rate'] = 8.0
    params['ext_nodes']   = np.array(
        [1600, 1500, 
         2100, 1900, 
         2000, 1900, 
         2900, 2100])
    
    
    
    weight = get_weight(0.15, params)
    
    params['exc_weight'] = weight
    params['ext_weights'] = [weight]*8
    
    if params['calc_lfp']:
        params = prep_LFP_approximation(params)
    
    
    ## Write parameters to file so the network can read it in
    with open("params", 'wb') as f:
        pickle.dump(params, f)
    return params

def prep_LFP_approximation(params):
    # class NetworkCell parameters:
    params['cellParameters'] = dict(
        # morphology='BallAndStick.hoc',  # set by main simulation
        templatefile='BallAndSticksTemplate.hoc',
        templatename='BallAndSticksTemplate',
        custom_fun=[set_active],
        custom_fun_args=[dict(Vrest=-65.)],  # [dict(Vrest=Vrest)] set at runtime
        templateargs=None,
        delete_sections=False,
    )
    
    
    # class NetworkPopulation parameters:
    params['populationParameters'] = dict(
        Cell=NetworkCell,
        cell_args=params['cellParameters'],
        pop_args=dict(
            radius=150.,  # population radius
            loc=0.,  # population center along z-axis
            scale=75.),  # standard deviation along z-axis
        rotation_args=dict(x=0., y=0.))
    
    # class Network parameters:
    params['networkParameters'] = dict(
        v_init=-70.,  # initial membrane voltage for all cells (mV)
        celsius=34,  # simulation temperature (deg. C)
        # OUTPUTPATH=OUTPUTPATH  # set in main simulation script
    )
    
    # class RecExtElectrode parameters:
    params['electrodeParameters'] = dict(
        x=np.zeros(16),  # x-coordinates of contacts
        y=np.zeros(16),  # y-coordinates of contacts
        z=np.linspace(1000., -250., 16),  # z-coordinates of contacts
        N=np.array([[0., 1., 0.] for _ in range(16)]),  # contact surface normals
        r=5.,  # contact radius
        n=100,  # n-point averaging for potential on each contact
        sigma=0.3,  # extracellular conductivity (S/m)
        method="linesource"  # use line sources
    )
    
    # class LaminarCurrentSourceDensity parameters:
    params['csdParameters'] = dict(
        z=np.c_[params['electrodeParameters']['z'] - 50.,
                params['electrodeParameters']['z'] + 50.],  # lower and upper boundaries
        r=np.array([params['populationParameters']['pop_args']['radius']] * 13)  # radius
    )
    
    ## TODO: Possibly remove?
    # method Network.simulate() parameters:
    params['networkSimulationArguments'] = dict(
        rec_pop_contributions=True,  # store contributions by each population
        to_memory=True,  # simulate to memory
        to_file=False  # simulate to file
    )
    
    # population names, morphologies, sizes and connection probability:
    params['morphologies'] = [
                    'BallAndSticks_E.hoc', 'BallAndSticks_I.hoc',
                    'BallAndSticks_E.hoc', 'BallAndSticks_I.hoc',
                    'BallAndSticks_E.hoc', 'BallAndSticks_I.hoc',
                    'BallAndSticks_E.hoc', 'BallAndSticks_I.hoc'
                    ]
    ## TODO: Remove
    # if TESTING:
    #     population_sizes = [32, 8]
    #     connectionProbability = [[1., 1.], [1., 1.]]
    # else:
    #     population_sizes = params['num_neurons']
    #     connectionProbability = params['connectivity']
    
    # synapse model. All corresponding parameters for weights,
    # connection delays, multapses and layerwise positions are
    # set up as shape (2, 2) nested lists for each possible
    # connection on the form:
    # [["E:E", "E:I"],
    #  ["I:E", "I:I"]].
    # using convention "<pre>:<post>"
    ## TODO: The model needs to be exported but can't be pickled (or does it?)
    # params['synapseModel'] = neuron.h.Exp2Syn
    # synapse parameters in terms of rise- and decay time constants
    # (tau1, tau2 [ms]) and reversal potential (e [mV])
    
    E2E = dict(tau1=0.2, tau2=1.8, e=0.)
    E2I = dict(tau1=0.2, tau2=1.8, e=0.)
    I2E = dict(tau1=0.1, tau2=9.0, e=-80.)
    I2I = dict(tau1=0.1, tau2=9.0, e=-80.)
    
    params['synapseParameters'] = [[E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I]]
    # synapse max. conductance (function, mean, st.dev., min.):
    params['weightFunction'] = np.random.normal
    # weight_<post><pre> values set by parameters file via main simulation scripts
    # weightArguments = [[dict(loc=weight_EE, scale=weight_EE / 10),
    #                     dict(loc=weight_IE, scale=weight_IE / 10)],
    #                    [dict(loc=weight_EI, scale=weight_EI / 10),
    #                     dict(loc=weight_II, scale=weight_II / 10)]]
    params['minweight'] = 0.  # weight values below this value will be redrawn
    
    # conduction delay (function, mean, st.dev., min.) using truncated normal
    # continuous random variable:
    params['delayFunction'] = st.truncnorm
    
    E2E = dict(a=(0.3 - 1.5) / 0.3, b=np.inf, loc=1.5, scale=0.3)
    E2I = dict(a=(0.3 - 1.4) / 0.4, b=np.inf, loc=1.4, scale=0.4)
    I2E = dict(a=(0.3 - 1.4) / 0.5, b=np.inf, loc=1.3, scale=0.5)
    I2I = dict(a=(0.3 - 1.2) / 0.6, b=np.inf, loc=1.2, scale=0.6)
    
    params['delayArguments'] = [[E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I]]
    # will be deprecated; has no effect with delayFunction = st.truncnorm:
    params['mindelay'] = None
    
    
    # Distributions of multapses. They are here defined via a truncated normal
    # continous random variable distribution which will be used to compute a
    # discrete probability distribution for integer numbers of
    # synapses 1, 2, ..., 100, via a scipy.stats.rv_discrete instance
    params['multapseFunction'] = st.truncnorm
    
    E2E = dict(a=(1 - 2.) / .4,
            b=(10 - 2.) / .4,
            loc=2.,
            scale=.4)
    E2I = dict(a=(1 - 2.) / .6,
              b=(10 - 2.) / .6,
              loc=2.,
              scale=.6)
    I2E = dict(a=(1 - 5.) / 0.9,
              b=(10 - 5.) / 0.9,
              loc=5.,
              scale=0.9)
    I2I = dict(a=(1 - 5.) / 1.1,
              b=(10 - 5.) / 1.1,
              loc=5.,
              scale=1.1)
    

    params['multapseArguments'] = [[E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I],
                                    [E2E, E2I, E2E, E2I, E2E, E2I, E2E, I2I]]
    
    
    # method NetworkCell.get_rand_idx_area_and_distribution_norm
    # parameters for layerwise synapse positions:
    
    ## Laminar thickness of layers in the microcircuit (in micrometers):
    ## Layer 1  : 90.
    ## Layer 2/3: 370.
    ## Layer 4  : 460.
    ## Layer 5  : 170.
    ## Layer 6  : 160.
    
    syn_pos = [[]]*8
    ## Create synapse positional parameters
    for i in range(len(params['label'])):
        for j in range(len(params['label'])):
            if params['label'][i] == "E" and params['label'][j] == "E":
                syn_pos[i].append(dict(section=['apic', 'dend'], ## E2E
                      fun=[st.norm, st.norm],
                      funargs=[dict(loc=0., scale=100.),
                              dict(loc=500., scale=100.)],
                      funweights=[0.5, 1.]
                      ))
            elif params['label'][i] == "E" and params['label'][j] != "E":
                syn_pos[i].append(dict(section=['apic', 'dend'],           ##E2I
                      fun=[st.norm],
                      funargs=[dict(loc=50., scale=100.)],
                      funweights=[1.]
                      ))
            elif params['label'][i] != "E" and params['label'][j] == "E":
                syn_pos[i].append(dict(section=['soma', 'apic', 'dend'],   ##I2E
                      fun=[st.norm],
                      funargs=[dict(loc=-50., scale=100.)],
                      funweights=[1.]
                      ))
            else:
                syn_pos[i].append(dict(section=['soma', 'apic', 'dend'],    ## I2I
                      fun=[st.norm],
                      funargs=[dict(loc=-100., scale=100.)],
                      funweights=[1.]
                      ))
    

    params['synapsePositionArguments'] = syn_pos
    
    # Parameters for extrinsic (e.g., cortico-cortical connections) synapses
    # and mean interval (ms)
    params['extSynapseParameters'] = dict(
        syntype='Exp2Syn',
        weight=0.0002,
        tau1=0.2,
        tau2=1.8,
        e=0.
    )
    params['netstim_interval'] = 25.
    
    
    
    return params

