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
        'num_neurons': np.array([400, 100, 400, 100]),
        'cell_params'  : [{
                            'V_m'        :  -70.,
                            'V_th'       :  -50.,
                            'V_reset'    :  -70.,
                            'C_m'        :  250.,
                            't_ref'      :    2.,
                            'tau_syn_ex' :   0.7,
                            'tau_syn_in' :   .25,
                            'E_L'        : -65.0,
                            'tau_m'      :  27.0,
                        },
                        {
                            'V_m'        :  -70.,
                            'V_th'       :  -49.,
                            'V_reset'    :  -70.,
                            'C_m'        :  250.,
                            't_ref'      :    2.,
                            'tau_syn_ex' :   0.9,
                            'tau_syn_in' :    .25,
                            'E_L'        : -65.0,
                            'tau_m'      :  23.0,
                        },
                         {
                            'V_m'        :  -70.,
                            'V_th'       :  -50.,
                            'V_reset'    :  -70.,
                            'C_m'        :  250.,
                            't_ref'      :    2.,
                            'tau_syn_ex' :   1.3,
                            'tau_syn_in' :   .35,
                            'E_L'        : -65.0,
                            'tau_m'      :  28.0,
                        },
                        {
                            'V_m'        :  -70.,
                            'V_th'       :  -49.,
                            'V_reset'    :  -70.,
                            'C_m'        :  250.,
                            't_ref'      :    2.,
                            'tau_syn_ex' :   1.2,
                            'tau_syn_in' :   .5,
                            'E_L'        : -65.0,
                            'tau_m'      :  24.0,
                        }],
        ## Increase in tau_syn_ex -> Increase in frequency
        
        'pop_name'   : ['L1_E', 'L1_I',
                        'L2_E' , 'L2_I'],
        'connection_strength': 0.0
        }
    ## AI Regime
    # {
    #                       'V_m'        :  -70.,
    #                       'V_th'       :  -50.,
    #                       'V_reset'    :  -76.,
    #                       'C_m'        :  250.,
    #                       't_ref'      :    3.,
    #                       'tau_syn_ex' :     1,
    #                       'tau_syn_in' :     1,
    #                       'E_L'        : -65.0,
    #                       'tau_m'      :  20.0,
    #                   },
    #                   {
    #                       'V_m'        :  -70.,
    #                       'V_th'       :  -49.,
    #                       'V_reset'    :  -76.,
    #                       'C_m'        :  250.,
    #                       't_ref'      :    4.,
    #                       'tau_syn_ex' :     1,
    #                       'tau_syn_in' :     1,
    #                       'E_L'        : -65.0,
    #                       'tau_m'      :  20.0,
    #                   },
    ###########################################################################
    ## Gamma/Beta Oscillations
    # [{
    #                     'V_m'        :  -70.,
    #                     'V_th'       :  -50.,
    #                     'V_reset'    :  -70.,
    #                     'C_m'        :  250.,
    #                     't_ref'      :    2.,
    #                     'tau_syn_ex' :   0.95,
    #                     'tau_syn_in' :   .25,
    #                     'E_L'        : -65.0,
    #                     'tau_m'      :  27.0,
    #                 },
    #                 {
    #                     'V_m'        :  -70.,
    #                     'V_th'       :  -49.,
    #                     'V_reset'    :  -70.,
    #                     'C_m'        :  250.,
    #                     't_ref'      :    2.,
    #                     'tau_syn_ex' :   0.9,
    #                     'tau_syn_in' :    .25,
    #                     'E_L'        : -65.0,
    #                     'tau_m'      :  23.0,
    #                 },
    #                  {
    #                     'V_m'        :  -70.,
    #                     'V_th'       :  -50.,
    #                     'V_reset'    :  -70.,
    #                     'C_m'        :  250.,
    #                     't_ref'      :    2.,
    #                     'tau_syn_ex' :   1.3,
    #                     'tau_syn_in' :   .25,
    #                     'E_L'        : -65.0,
    #                     'tau_m'      :  28.0,
    #                 },
    #                 {
    #                     'V_m'        :  -70.,
    #                     'V_th'       :  -49.,
    #                     'V_reset'    :  -70.,
    #                     'C_m'        :  250.,
    #                     't_ref'      :    2.,
    #                     'tau_syn_ex' :   1.3,
    #                     'tau_syn_in' :   .5,
    #                     'E_L'        : -65.0,
    #                     'tau_m'      :  24.0,
    #                 }]
    
    ################################################################
    ## Specify connectivity in and between layers and populations ##
    ################################################################
    
    # Connectivity matrix
    k = params['connection_strength']               
    params['connectivity'] = np.array(
            [[0.1 , 0.25, k    , k   ],
             [0.5 , 0.25, k    , k   ],
             [k   , k   , 0.1  , 0.3 ],
             [k   , k   , 0.45 , 0.25]]
            )
    
    ################################
    ## Specify synapse properties ##
    ################################
    
    params['syn_type'] = np.array([["E", "E", "E", "E"],
                                   ["I", "I", "I", "I"],
                                   ["E", "E", "E", "E"],
                                   ["I", "I", "I", "I"]])
    
    
    syn_strength = 8
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

    params['ext_rate'] = 25.0
    params['ext_nodes']   = np.array(
        [1000, 1000, 
         1000, 1000])
    
    weight = syn_strength
    
    params['exc_weight'] = weight
    params['ext_weights'] = [weight]*8
    
    ## Write parameters to file so the network can read it in
    with open("params", 'wb') as f:
        pickle.dump(params, f)
    return params