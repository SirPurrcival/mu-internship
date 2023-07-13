import numpy as np
import pickle

def setup():
    #######################################
    ## Set parameters for the simulation ##
    #######################################
    ## Recording and simulation parameters
    params = {
        'rec_start'  :    200.,                                                # start point for data recording
        'rec_stop'   :   2000.,                                                # end points for data recording
        'record_to'  : 'memory',
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
        'second_net' :    True,
        'deep_input' :       0,
        'num_neurons': np.array([400, 100, 400, 100]),
        'sd'         :  0.1,
        'cell_params_net1'  : [{
                            'tau_syn_ex' :   1.0,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  19.0
                        },
                        {
                            'tau_syn_ex' :   0.5,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  20.0
                        },
                         {
                            'tau_syn_ex' :   0.8,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  18.0
                        },
                        {
                            'tau_syn_ex' :   0.5,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  20.0
                        }],
        'cell_params_net2'  : [{
                            'tau_syn_ex' :   1.0,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  19.0
                        },
                        {
                            'tau_syn_ex' :   0.5,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  20.0
                        },
                         {
                            'tau_syn_ex' :   0.8,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  18.0
                        },
                        {
                            'tau_syn_ex' :   0.5,
                            'tau_syn_in' :   0.5,
                            'tau_m'      :  20.0
                        }],
        
        'pop_name'   : ['L1_E', 'L1_I',
                        'L2_E' , 'L2_I'],
        'interlaminar_connections': 0.1,
        'E/I ratio'               :  2., 
        'intercircuit_strength'   : 0.05
    }
    
    ck = params['intercircuit_strength']
    
    params['net1_net2_connections'] = np.array(
        ##            Target
        ##    Net2_E1        Net2_I1       Net2_E2       Net2_I2 
            [[ck           , ck          , 0.0         , 0.0         ], ## Net1_E1
             [0.0          , 0.0         , 0.0         , 0.0         ], ## Net1_I1   Source
             [0.0          , 0.0         , 0.0         , 0.0         ], ## Net1_E2
             [0.0          , 0.0         , 0.0         , 0.0         ]] ## Net1_I2
            )
    params['net2_net1_connections'] = np.array(
        ##            Target
        ##    Net1_E1        Net1_I1       Net1_E2       Net1_I2 
            [[ck           , ck          , 0.0         , 0.0         ], ## Net2_E1
             [0.0          , 0.0         , 0.0         , 0.0         ], ## Net2_I1   Source
             [0.0          , 0.0         , 0.0         , 0.0         ], ## Net2_E2
             [0.0          , 0.0         , 0.0         , 0.0         ]] ## Net2_I2
            )
    
    ################################################################
    ## Specify connectivity in and between layers and populations ##
    ################################################################
    
    # Connectivity matrix
    k = params['interlaminar_connections']
    ei = params['E/I ratio']               
    params['intralaminar'] = np.array(
        ##            Target
        ##    E1              I1                E2           I2 
            [[0.1          , 0.25        , 0.          , 0.           ], ## E1
             [0.5          , 0.25        , 0.          , 0.           ], ## I1   Source
             [0.           , 0.          , 0.1         , 0.25         ], ## E2
             [0.           , 0.          , 0.5         , 0.25        ]] ## I2
            )
    params['interlaminar'] = np.array(
        ##            Target
        ##    E1              I1                E2           I2 
            [[0.           , 0.          , k*(1-(1/ei)), k*(1-(1/ei))], ## E1
             [0.           , 0.          , k*(1/ei)    , k*(1/ei)    ], ## I1   Source
             [k*(1-(1/ei)) , k*(1-(1/ei)), 0.          , 0.          ], ## E2
             [k*(1/ei)     , k*(1/ei)    , 0.          , 0.          ]] ## I2
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
        [100, 30, 
         100, 30])
    
    weight = syn_strength
    
    params['exc_weight'] = weight
    params['ext_weights'] = [weight]*8
    
    ## Write parameters to file so the network can read it in
    with open("params", 'wb') as f:
        pickle.dump(params, f)
    return params