import os
import numpy as np
import matplotlib.pyplot as plt
import neuron
import nest 
from copy import deepcopy

import example_network_methods as methods
from pynestml.frontend.pynestml_frontend import generate_nest_target
from lfpykernels import KernelApprox, GaussCylinderPotential, KernelApproxCurrentDipoleMoment


def prep_LFP_kernel(params):

    plt.rcParams.update({
        'axes.xmargin': 0.01,
        'axes.ymargin': 0.01,
    })
    
    # recompile mod files if needed
    mech_loaded = neuron.load_mechanisms('mod')
    if not mech_loaded:
        os.system('cd mod && nrnivmodl && cd -')
        mech_loaded = neuron.load_mechanisms('mod')
    print(f'mechanisms loaded: {mech_loaded}')
    
    
    # parameters
    pset = dict(
        weight_EE=0.00015,  # E to E connection weight
        weight_IE=0.00012,  # E to I
        weight_EI=0.0045,   # I to E
        weight_II=0.0020,   # I to I
        weight_scaling=1.,  # global weight scaling
        biophys='frozen',   # passive-frozen biophysics
        g_eff=True          # if True, account for the effective membrane time constants
    )
    
    TRANSIENT = 200  # ignore 1st 200 ms of simulation in analyses
    dt = params['resolution']
    tau = 50  # time lag relative to spike for kernel predictions
    
    ## TODO: Adapt this. Possible solution:
    ## Run simulation for 200ms with spike recorders, get average firing rates
    ## per population, then approximate LFPs with firing rates later
    # Assumed average firing rate of presynaptic populations X
    mean_nu_X = dict(L1_Htr3a=2.3, 
                     L23_E=4.9,
                     L23_Pvalb=4.,
                     L23_Sst=4.,
                     L23_Htr3a=4.,
                     L4_E=4.,
                     L4_Pvalb=4.,
                     L4_Sst=4.,
                     L4_Htr3a=4.,
                     L5_E=4.,
                     L5_Pvalb=4.,
                     L5_Sst=4.,
                     L5_Htr3a=4.,
                     L6_E=4.,
                     L6_Pvalb=4.,
                     L6_Sst=4.,
                     L6_Htr3a=4.
                     )  # spikes/s
    
    # assumed typical postsynaptic potential for each population
    Vrest = -70. 
    
    # presynaptic activation time
    t_X = TRANSIENT
    
    # Compile and install NESTML FIR_filter.nestml model
    nestml_model_file = 'FIR_filter.nestml'
    nestml_model_name = 'fir_filter_nestml'
    target_path = '/tmp/fir-filter'
    logging_level = 'INFO'
    module_name = 'nestmlmodule'
    store_log = False
    suffix = '_nestml'
    dev = True
    
    input_path = os.path.join(os.path.realpath(nestml_model_file))
    nest_path = nest.ll_api.sli_func("statusdict/prefix ::")
    generate_nest_target(input_path=input_path,
                          target_path=target_path,
                          suffix=suffix)
    
    # to_nest(input_path, target_path, logging_level, module_name, store_log, suffix, dev)
    # install_nest(target_path, nest_path)
    
    nest.set_verbosity("M_ALL")
    
    ## If it's already loaded ignore the error and continue
    try:
        nest.Install(module_name)
    except:
        pass
    
    nest.ResetKernel()
    
    # create kernels from multicompartment neuron network description
                    
    # kernel container
    H_YX = dict()
    
    # define biophysical membrane properties
    if pset['biophys'] == 'frozen':
        set_biophys = [methods.set_frozen_hay2011, methods.make_cell_uniform]
    elif pset['biophys'] == 'lin':
        set_biophys = [methods.set_Ih_linearized_hay2011, methods.make_cell_uniform]
    else:
        raise NotImplementedError
    
    # synapse max. conductance (function, mean, st.dev., min.):
    E2E = pset['weight_EE']
    E2I = pset['weight_EI']
    I2E = pset['weight_IE']
    I2I = pset['weight_II']
    
    weights = [[I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I, I2E, I2I, I2I, I2I],
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
    
    # Not using RecExtElectrode class as we anyway average potential in
    # space for each source element. This is to replaced by
    # a closed form volumetric method (point source & volumetric contacts
    # should result in same mappings as volumetric source & point contacs)
    
    # Predictor assuming planar disk source elements convolved with Gaussian
    # along z-axis
    gauss_cyl_potential = GaussCylinderPotential(
        cell=None,
        z=params['electrodeParameters']['z'],
        sigma=params['electrodeParameters']['sigma'],
        R=params['populationParameters']['pop_args']['radius'],
        sigma_z=params['populationParameters']['pop_args']['scale'],
        )
    
    # set up recording of current dipole moments. 
    # The class KernelApproxCurrentDipoleMoment accounts only for contributions
    # along the vertical z-axis as other components cancel with rotational symmetry
    current_dipole_moment = KernelApproxCurrentDipoleMoment(cell=None)
    
    
    # compute kernels for each pair of pre- and postsynaptic populations.
    # Iterate over presynaptic populations
    for i, (X, N_X) in enumerate(zip(params['layer_type'],
                                     [int(x) for x in params['num_neurons']*params['N_scale']])):
        # iterate over postsynaptic populations
        for j, (Y, N_Y, morphology) in enumerate(zip(params['layer_type'],
                                                     [int(x) for x in params['num_neurons']*params['N_scale']],
                                                     params['morphologies'])):
            # set up LFPy.NetworkCell parameters for postsynaptic units
            cellParameters = deepcopy(params['cellParameters'])
            cellParameters.update(dict(
                morphology=morphology,
                custom_fun=set_biophys,
                custom_fun_args=[dict(Vrest=Vrest), dict(Vrest=Vrest)],
            ))
    
            # some inputs must be lists
            synapseParameters = [
                dict(weight=weights[ii][j],
                     syntype='Exp2Syn',
                     **params['synapseParameters'][ii][j])
                for ii in range(len(params['layer_type']))]
            synapsePositionArguments = [
                params['synapsePositionArguments'][ii][j]
                for ii in range(len(params['layer_type']))]
    
            # Create KernelApprox object
            kernel = KernelApprox(
                X=params['layer_type'],
                Y=Y,
                N_X=np.array([int(x) for x in params['num_neurons']*params['N_scale']]),
                N_Y=N_Y,
                C_YX=np.array(params['connectivity'][i]),
                cellParameters=cellParameters,
                populationParameters=params['populationParameters']['pop_args'],
                multapseFunction=params['multapseFunction'],
                multapseParameters=[params['multapseArguments'][ii][j] for ii in range(len(params['layer_type']))],
                delayFunction=params['delayFunction'],
                delayParameters=[params['delayArguments'][ii][j] for ii in range(len(params['layer_type']))],
                synapseParameters=synapseParameters,
                synapsePositionArguments=synapsePositionArguments,
                extSynapseParameters=params['extSynapseParameters'],
                nu_ext=1000. / params['netstim_interval'],
                n_ext=params['ext_nodes'][j],
                nu_X=mean_nu_X,
            )
            print(f"Iteration {i} {j}")
            # make kernel predictions and update container dictionary
            H_YX['{}:{}'.format(Y, X)] = kernel.get_kernel(
                probes=[gauss_cyl_potential, current_dipole_moment],
                Vrest=Vrest, dt=dt, X=X, t_X=t_X, tau=tau,
                g_eff=pset['g_eff'],
                fir=True
            )
    return H_YX