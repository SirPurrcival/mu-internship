import numpy as np
import pickle

def setup():
    #######################################
    ## Set parameters for the simulation ##
    #######################################
    
    ## Recording and simulation parameters
    params = {
        'rec_start'  :    200.,                                                # start point for data recording
        'rec_stop'   :   1000.,                                                # end points for data recording
        'sim_time'   :   1000.,                                                # Time the network is simulated in ms
        'calc_lfp'   :   False,                                                # Flag to use LFP approximation procedure
        'verbose'    :    True,                                                # Flag for verbose function output
        'K_scale'    :      1.,                                                # Scaling factor for connections
        'syn_scale'  :      1.,                                                # Scaling factor for synaptic strenghts
        'N_scale'    :     0.1,                                                # Scaling factor for the number of neurons
        'R_scale'    :     0.1,                                                # Fraction of neurons to be recorded from
        'opt_run'    :   False,                                                # Flag for optimizer run, run minimal settings
        'g'          :     -4.,                                                # Excitation-Inhibition balance
        'resolution' :     0.1,                                                # Resolution of the simulaton
        'transient'  :     200,                                                # Ignore the first x ms of the simulation
        'th_in'      :     20.,                                                # Thalamic input, nodes x frequency
        'th_start'   :    400.,
        'th_stop'    :    600.,
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
                        'tau_m' :     10.0,},
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
             [0.0077, 0.0059, 0.0497, 0.135 , 0.0067, 0.0003, 0.0453, 0.    ],
             [0.0691, 0.0029, 0.0794, 0.1597, 0.0033, 0.    , 0.1057, 0.    ],
             [0.1004, 0.0622, 0.0505, 0.0057, 0.0831, 0.3726, 0.0204, 0.    ],
             [0.0548, 0.0269, 0.0257, 0.0022, 0.06  , 0.3158, 0.0086, 0.    ],
             [0.0156, 0.0066, 0.0211, 0.0166, 0.0572, 0.0197, 0.0396, 0.2252],
             [0.0364, 0.001 , 0.0034, 0.0005, 0.0277, 0.008 , 0.0658, 0.1443]]
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
    
    ## Write parameters to file so the network can read it in
    with open("params", 'wb') as f:
        pickle.dump(params, f)
    return params