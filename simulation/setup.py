import numpy as np
import pickle

## Make E/I ratio
## Make total connections value (percentage)

## Figure out gamma range
## Figure out beta range

## spectral density plot

def setup():
    #######################################
    ## Set parameters for the simulation ##
    #######################################
    
    ## General parameters
    settings  = {
        'rec_start'  :    200.,                                                # start point for data recording
        'rec_stop'   :   1000.,                                                # end points for data recording
        'record_to'  :'memory',
        'sim_time'   :   1000.,                                                # Time the network is simulated in ms
        'calc_lfp'   :   False,                                                # Flag to use LFP approximation procedure
        'verbose'    :    True,                                                # Flag for verbose function output
        'K_scale'    :      1.,                                                # Scaling factor for connections
        'syn_scale'  :      1.,                                                # Scaling factor for synaptic strenghts
        'N_scale'    :      1.,                                                # Scaling factor for the number of neurons
        'R_scale'    :      1.,                                                # Fraction of neurons to be recorded from
        'opt_run'    :   False,                                                # Flag for optimizer run, run minimal settings
        'g'          :     -4.,                                                # Excitation-Inhibition balance
        'resolution' :     0.1,                                                # Resolution of the simulaton
        'transient'  :     200,                                                # Ignore the first x ms of the simulation
        'second_net' :   False
        }
    
    ## Parameters for network 1
    net1_dict = {
        'th_in'      :      0.,                                                # Thalamic input in Hz
        'th_start'   :    500.,
        'th_stop'    :    510.,
        'num_neurons': np.array([400, 100, 400, 100]),
        'cell_params'  : [{
                            'tau_syn_ex' :   1.4,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  19.0
                        },
                        {
                            'tau_syn_ex' :    1.,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  19.0
                        },
                         {
                            'tau_syn_ex' :   1.1,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  13.0
                        },
                        {
                            'tau_syn_ex' :    1.,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  15.0
                        }],
        
        'pop_name'   : ['L1_E', 'L1_I',
                        'L2_E' , 'L2_I'],
        }
    
    ## Parameters for network 2
    net2_dict = net1_dict
    ## Recording and simulation parameters
    params = {
        'rec_start'  :    200.,                                                # start point for data recording
        'rec_stop'   :   2000.,                                                # end points for data recording
        'record_to'  :'memory',
        'sim_time'   :   2000.,                                                # Time the network is simulated in ms
        'calc_lfp'   :   False,                                                # Flag to use LFP approximation procedure
        'verbose'    :    True,                                                # Flag for verbose function output
        'K_scale'    :      1.,                                                # Scaling factor for connections
        'syn_scale'  :      1.,                                                # Scaling factor for synaptic strenghts
        'N_scale'    :      1.,                                                # Scaling factor for the number of neurons
        'R_scale'    :      1.,                                                # Fraction of neurons to be recorded from
        'opt_run'    :   False,                                                # Flag for optimizer run, run minimal settings
        'g'          :     -4.,                                                # Excitation-Inhibition balance
        'resolution' :     0.1,                                                # Resolution of the simulaton
        'transient'  :     200,                                                # Ignore the first x ms of the simulation
        'th_in'      :      0.,                                                # Thalamic input in Hz
        'th_start'   :    500.,
        'th_stop'    :    510.,
        'second_net' :   False,
        'num_neurons': np.array([400, 100, 400, 100]),
        'cell_params'  : [{
                            'tau_syn_ex' :   1.2,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  14.0
                        },
                        {
                            'tau_syn_ex' :   0.5,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  25.0
                        },
                         {
                            'tau_syn_ex' :   1.1,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  14.0
                        },
                        {
                            'tau_syn_ex' :   0.5,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  25.0
                        }],
        
        'pop_name'   : ['L1_E', 'L1_I',
                        'L2_E' , 'L2_I'],
        'interlaminar_connections': 0.0,
        'E/I ratio'               :  4., 
        'net1_net2_connections': np.array(
            ##            Target
            ##    Net2_E1        Net2_I1       Net2_E2       Net2_I2 
                [[0.1          , 0.1         , 0.1         , 0.1         ], ## Net1_E1
                 [0.1          , 0.1         , 0.1         , 0.1         ], ## Net1_I1   Source
                 [0.1          , 0.1         , 0.1         , 0.1         ], ## Net1_E2
                 [0.1          , 0.1         , 0.1         , 0.1         ]] ## Net1_I2
                ),
        'net2_net1_connections': np.array(
            ##            Target
            ##    Net1_E1        Net1_I1       Net1_E2       Net1_I2 
                [[0.1          , 0.1         , 0.1         , 0.1         ], ## Net2_E1
                 [0.1          , 0.1         , 0.1         , 0.1         ], ## Net2_I1   Source
                 [0.1          , 0.1         , 0.1         , 0.1         ], ## Net2_E2
                 [0.1          , 0.1         , 0.1         , 0.1         ]] ## Net2_I2
                )
        }
    
    ################################################################
    ## Specify connectivity in and between layers and populations ##
    ################################################################
    
    # Connectivity matrix
    k = params['interlaminar_connections']
    ei = params['E/I ratio']               
    params['connectivity'] = np.array(
        ##            Target
        ##    E1              I1                E2           I2 
            [[0.1          , 0.25        , k*(1-(1/ei)), k*(1-(1/ei))], ## E1
             [0.5          , 0.25        , k*(1/ei)    , k*(1/ei)    ], ## I1   Source
             [k*(1-(1/ei)) , k*(1-(1/ei)), 0.1         , 0.25        ], ## E2
             [k*(1/ei)     , k*(1/ei)    , 0.5         , 0.25        ]] ## I2
            )
    
    ################################
    ## Specify synapse properties ##
    ################################
    
    params['syn_type'] = np.array([["E", "E", "E", "E"],
                                   ["I", "I", "I", "I"],
                                   ["E", "E", "E", "E"],
                                   ["I", "I", "I", "I"]])
    
    
    syn_strength = 80
    ## Synaptic strength
    params['syn_strength'] = np.array([
                                    [syn_strength]*4,
                                    [syn_strength*params['g']]*4,
                                    [syn_strength]*4,
                                    [syn_strength*params['g']]*4]
                                    )
    
    #######################################
    ## Background stimulation parameters ##
    #######################################

    params['ext_rate'] = 30.0
    params['ext_nodes']   = np.array(
        [100, 60, 
         100, 60])
    
    weight = syn_strength
    
    params['exc_weight'] = weight
    params['ext_weights'] = [weight]*8
    
    ## Write parameters to file so the network can read it in
    with open("params", 'wb') as f:
        pickle.dump(params, f)
    return params