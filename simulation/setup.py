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
        'calc_lfp'   :  False,                                                      # Flag to use LFP approximation procedure
        'verbose'    :  True,                                                       # Flag for verbose function output
        'K_scale'    :     1.,                                                      # Scaling factor for connections
        'syn_scale'  :     1.,                                                      # Scaling factor for synaptic strenghts
        'N_scale'    :     .5,                                                      # Scaling factor for the number of neurons
        'R_scale'    :     0.1,                                                     # Fraction of neurons to be recorded from
        'opt_run'    :   False,                                                     # Flag for optimizer run, run minimal settings
        'g'          :      4.,                                                     # Excitation-Inhibition balance
        'resolution' :   2**-3,                                                     # Resolution of the simulaton
        'transient'  :     200,                                                     # Ignore the first x ms of the simulation
        'th_in'      :   100.0*0.0,                                                 # Thalamic input, nodes x frequency
        'num_neurons': np.array([776, 47386, 3876, 2807, 6683, 70387, 9502, 5455,   # Number of neurons by population
                          2640, 20740, 2186, 1958, 410, 19839, 1869, 1869, 325]),
        'label'      : ['Htr','E','Pv','Sst','Htr','E','Pv','Sst','Htr','E','Pv',   # Label for the populations
                        'Sst','Htr','E','Pv','Sst','Htr'],
        'cell_type'  : np.load('cells.npy', allow_pickle=True).item(),              # cell types, courtesy of the allen institute
        'layer_type' : ['L1_Htr3a',                                                 # Layer and cell types
                        'L23_E', 'L23_Pvalb', 'L23_Sst', 'L23_Htr3a',
                        'L4_E', 'L4_Pvalb', 'L4_Sst', 'L4_Htr3a',  
                        'L5_E', 'L5_Pvalb', 'L5_Sst', 'L5_Htr3a', 
                        'L6_E', 'L6_Pvalb',  'L6_Sst', 'L6_Htr3a']
        }
    
    ################################################################
    ## Specify connectivity in and between layers and populations ##
    ################################################################
    
    # Connectivity matrix layertype X layertype
    params['connectivity'] = np.array([[0.656, 0.356, 0.093, 0.068, 0.4644, 0.148, 0    , 0    , 0    , 0.148, 0    , 0    , 0    , 0.148, 0    , 0    , 0    ],
                                      [0    , 0.16 , 0.395, 0.182, 0.105 , 0.016, 0.083, 0.083, 0.083, 0.083, 0.081, 0.102, 0    , 0    , 0    , 0    , 0    ],
                                      [0.024, 0.411, 0.451, 0.03 , 0.22  , 0.05 , 0.05 , 0.05 , 0.05 , 0.07 , 0.073, 0    , 0    , 0    , 0    , 0    , 0    ],
                                      [0.279, 0.424, 0.857, 0.082, 0.77  , 0.05 , 0.05 , 0.05 , 0.05 , 0.021, 0    , 0    , 0    , 0    , 0    , 0    , 0    ],
                                      [0    , 0.087, 0.02 , 0.625, 0.028 , 0.05 , 0.05 , 0.05 , 0.05 , 0    , 0    , 0    , 0    , 0    , 0    , 0    , 0    ],
                                      [0    , 0.14 , 0.100, 0.1  , 0.1   , 0.243, 0.43 , 0.571, 0.571, 0.104, 0.101, 0.128, 0.05 , 0.032, 0    , 0    , 0    ],
                                      [0    , 0.25 , 0.050, 0.05 , 0.05  , 0.437, 0.451, 0.03 , 0.22 , 0.088, 0.091, 0.03 , 0.03 , 0    , 0    , 0    , 0    ],
                                      [0.241, 0.25 , 0.050, 0.05 , 0.05  , 0.351, 0.857, 0.082, 0.77 , 0.026, 0.03 , 0    , 0.03 , 0    , 0    , 0    , 0    ],
                                      [0    , 0.25 , 0.050, 0.05 , 0.05  , 0.351, 0.02 , 0.625, 0.028, 0    , 0.03 , 0.03 , 0.03 , 0    , 0    , 0    , 0    ],
                                      [0.017, 0.021, 0.05 , 0.05 , 0.05  , 0.007, 0.05 , 0.05 , 0.05 , 0.116, 0.083, 0.063, 0.105, 0.047, 0.03 , 0.03 , 0.03 ],
                                      [0    , 0    , 0.102, 0    , 0     , 0    , 0.034, 0.03 , 0.03 , 0.455, 0.361, 0.03 , 0.22 , 0.03 , 0.01 , 0.01 , 0.01 ],
                                      [0.203, 0.169, 0    , 0.017, 0     , 0.056, 0.03 , 0.006, 0.03 , 0.317, 0.857, 0.04 , 0.77 , 0.03 , 0.01 , 0.01 , 0.01 ],
                                      [0    , 0    , 0    , 0    , 0     , 0.03 , 0.03 , 0.03 , 0.03 , 0.125, 0.02 , 0.625, 0.02 , 0.03 , 0.01 , 0.01 , 0.01 ],
                                      [0    , 0    , 0    , 0    , 0     , 0    , 0    , 0    , 0    , 0.012, 0.01 , 0.01 , 0.01 , 0.026, 0.145, 0.1  , 0.1  ],
                                      [0    , 0.1  , 0    , 0    , 0     , 0.1  , 0    , 0    , 0    , 0.1  , 0.03 , 0.03 , 0.03 , 0.1  , 0.08 , 0.1  , 0.08 ],
                                      [0    , 0    , 0    , 0    , 0     , 0    , 0    , 0    , 0    , 0.03 , 0.03 , 0.03 , 0.03 , 0.1  , 0.05 , 0.05 , 0.05 ],
                                      [0    , 0    , 0    , 0    , 0     , 0    , 0    , 0    , 0    , 0.03 , 0.03 , 0.03 , 0.03 , 0.1  , 0.05 , 0.05 , 0.03 ]])
    
    ################################
    ## Specify synapse properties ##
    ################################
    
    params['syn_type'] = np.array([["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E"], 
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E"], 
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E"], 
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E","E"], 
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"],
                                      ["I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I","I"]])
    
    ## Synaptic strength
    params['syn_strength'] = np.array(
                                       [[1.73, 0.53, 0.48, 0.57, 0.78, 0.42, 0   , 0   , 0   , 0.42, 0   , 0   , 0   , 0.42, 0   , 0   , 0   ],
                                       [0      , 0.36, 1.49, 0.86, 1.31, 0.34, 1.39, 0.69, 0.91, 0.74, 1.32, 0.53, 0   , 0   , 0   , 0   , 0   ],
                                       [0.37 , 0.48, 0.68, 0.42, 0.41, 0.56, 0.68, 0.42, 0.41, 0.2 , 0.79, 0   , 0   , 0   , 0   , 0   , 0   ],
                                       [0.47 , 0.31, 0.5 , 0.15, 0.52, 0.3 , 0.5 , 0.15, 0.52, 0.22, 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                                       [0      , 0.28, 0.18, 0.32, 0.37, 0.29, 0.18, 0.32, 0.37, 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ],
                                       [0      , 0.78, 1.39, 0.69, 0.91, 0.83, 1.29, 0.51, 0.51, 0.63, 1.25, 0.52, 0.91, 0.96, 0   , 0   , 0   ],
                                       [0      , 0.56, 0.68, 0.42, 0.41, 0.64, 0.68, 0.42, 0.41, 0.73, 0.94, 0.42, 0.41, 0   , 0   , 0   , 0   ],
                                       [0.39 , 0.3 , 0.5 , 0.15, 0.52, 0.29, 0.5 , 0.15, 0.52, 0.28, 0.45, 0.28, 0.52, 0   , 0   , 0   , 0   ],
                                       [0      , 0.29, 0.18, 0.32, 0.37, 0.29, 0.18, 0.32, 0.37, 0   , 0.18, 0.33, 0.37, 0   , 0   , 0   , 0   ],
                                       [0.76 , 0.47, 1.25, 0.52, 0.91, 0.38, 1.25, 0.52, 0.91, 0.75, 1.2 , 0.52, 1.31, 0.4 , 2.5 , 0.52, 1.31],
                                       [0      , 0   , 0.51, 0   , 0   , 0   , 0.94, 0.42, 0.41, 0.81, 1.19, 0.41, 0.41, 0.81, 1.19, 0.41, 0.41],
                                       [0.31 , 0.25, 0   , 0.39, 0   , 0.28, 0.45, 0.28, 0.52, 0.27, 0.4 , 0.4 , 0.52, 0.27, 0.4 , 0.4 , 0.52],
                                       [0     , 0   , 0   , 0   , 0   , 0.29, 0.18, 0.33, 0.37, 0.28, 0.18, 0.33, 0.37, 0.28, 0.18, 0.33, 0.37],
                                       [0     , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.23, 2.5 , 0.52, 1.31, 0.94, 3.8 , 0.52, 1.31],
                                       [0     , 0.81, 0   , 0   , 0   , 0.81, 0   , 0   , 0   , 0.81, 1.19, 0.41, 0.41, 0.81, 1.19, 0.41, 0.41],
                                       [0     , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.27, 0.4 , 0.4 , 0.52, 0.27, 0.4 , 0.4 , 0.52],
                                       [0     , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0.28, 0.18, 0.33, 0.37, 0.28, 0.18, 0.33, 0.37]])
    
    
    #######################################
    ## Background stimulation parameters ##
    #######################################

    params['ext_rate'] = 1.0
    params['ext_nodes']   = np.array([2000, 
                                      1400, 2000, 1400, 1400, 
                                      1500, 2100, 2100, 1500, 
                                      1500, 1500, 1500, 1500, 
                                      1700, 1500, 1500, 1500])
    params['ext_weights'] = [3.8, 
                             7.6, 5.3, 4.99, 3.45, 
                             5.1, 4.9, 3.2, 3.15, #8.14, 6.6591 
                             5.6, 5.8, 2.35, 3.95, 
                             5.35, 3.5, 2.35, 7.09]
    
    ## Old 0.15 K values
    # [3.7965744951318477, 
    #                          7.762524932070633, 12.818597965224434, 5.947184199171211, 3.8553839597064073, 
    #                          2.95422324169895, 7.841405598126207, 4.897362276797087, 2.127810123924406, 
    #                          3.2981009467815937, 3.38233409130867667, 2.0180440299163823, 1.444171890699541, 
    #                          5.360764434899142, 4.92695472801618, 3.137543770653454, 3.366822560118546]
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
    params['morphologies'] = ['BallAndSticks_I.hoc', 
                    'BallAndSticks_E.hoc', 'BallAndSticks_I.hoc', 'BallAndSticks_I.hoc', 'BallAndSticks_I.hoc',
                    'BallAndSticks_E.hoc', 'BallAndSticks_I.hoc', 'BallAndSticks_I.hoc', 'BallAndSticks_I.hoc',
                    'BallAndSticks_E.hoc', 'BallAndSticks_I.hoc', 'BallAndSticks_I.hoc', 'BallAndSticks_I.hoc',
                    'BallAndSticks_E.hoc', 'BallAndSticks_I.hoc', 'BallAndSticks_I.hoc', 'BallAndSticks_I.hoc'
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
    
    params['synapseParameters'] = [[I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I]]
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
    
    params['delayArguments'] = [[I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I]]
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
    

    params['multapseArguments'] = [[I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I, E2E, E2I, E2I, E2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
                                    [I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I]]
    
    
    # method NetworkCell.get_rand_idx_area_and_distribution_norm
    # parameters for layerwise synapse positions:
    
    ## Laminar thickness of layers in the microcircuit (in micrometers):
    ## Layer 1  : 90.
    ## Layer 2/3: 370.
    ## Layer 4  : 460.
    ## Layer 5  : 170.
    ## Layer 6  : 160.
    
    syn_pos = [[]]*17
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

