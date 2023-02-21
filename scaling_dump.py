#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All the stuff that relates to scaling down the network. Code taken from:
https://github.com/INM-6/multi-area-model/blob/master/multiarea_model/multiarea_model.py
"""

import numpy as np

import json
import numpy as np
import os
from itertools import product
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

from config import base_path
from .default_params import complete_area_list, population_list
from nested_dict import nested_dict

def create_mask(structure, target_pops=population_list,
                source_pops=population_list,
                target_areas=complete_area_list,
                source_areas=complete_area_list,
                complete_area_list=complete_area_list,
                external=True,
                **keywords):
    """
    Create a mask for the connection matrices to filter
    for specific pairs of populations.
    Parameters
    ----------
    structure : dict
        Structure of the network. Define the populations for each single area.
    target_pops : list, optinal
        List of target populations for each target area in the mask to be created.
        Default to population_list defined in default_params.
    source_pops : list, optinal
        List of source populations for each source area in the mask to be created.
        Default to population_list defined in default_params.
    target_areas : list, optinal
        List of target areas in the mask to be created.
        Defaults to the complete_area_list defined in default_params.
    source_areas : list, optinal
        List of source areas in the mask to be created.
        Defaults to the complete_area_list defined in default_params.
    complete_area_list : list, optional
        List of areas in the network. Defines the order of areas
        in the given matrix. Defaults to the complete_area_list defined in default_params.
    external : bool, optional
        Whether to include input from external source in the mask.
        Defaults to True.
    cortico_cortical : bool, optional
        Whether to filter for cortico_cortical connections only.
        Defaults to False.
    internal : bool, optional
        Whether to filter for internal connections only.
        Defaults to False.
    """
    target_mask = create_vector_mask(structure, pops=target_pops,
                                     areas=target_areas, complete_area_list=complete_area_list)
    source_mask = create_vector_mask(structure, pops=source_pops,
                                     areas=source_areas, complete_area_list=complete_area_list)
    if external:
        source_mask = np.append(source_mask, np.array([True]))
    else:
        source_mask = np.append(source_mask, np.array([False]))
    mask = np.outer(target_mask, source_mask)

    if 'cortico_cortical' in keywords and keywords['cortico_cortical']:
        negative_mask = np.zeros_like(mask, dtype=np.bool)
        for source in source_areas:
            smask = create_mask(structure,
                                target_pops=population_list,
                                target_areas=[source], source_areas=[source],
                                source_pops=population_list,
                                external=True)
            negative_mask = np.logical_or(negative_mask, smask)
        mask = np.logical_and(np.logical_not(
            np.logical_and(mask, negative_mask)), mask)
    if 'internal' in keywords and keywords['internal']:
        negative_mask = np.zeros_like(mask, dtype=np.bool)
        for source in source_areas:
            smask = create_mask(structure,
                                target_pops=population_list,
                                target_areas=[source], source_areas=[source],
                                source_pops=population_list)
            negative_mask = np.logical_or(negative_mask, smask)
        mask = np.logical_and(mask, negative_mask)
    return 

def dict_to_matrix(d, area_list, structure):
    """
    Convert a dictionary containing connectivity
    information of a network defined by structure to a matrix.
    Parameters
    ----------
    d : dict
        Dictionary to be converted.
    area_list: list
        List of areas in the network. Defines the order of areas
        in the given matrix.
    structure : dict
        Structure of the network. Define the populations for each single area.
    """
    dim = 0
    for area in structure.keys():
        dim += len(structure[area])

    M = np.zeros((dim, dim + 1))
    i = 0
    for target_area in area_list:
        for target_pop in structure[target_area]:
            j = 0
            for source_area in area_list:
                for source_pop in structure[source_area]:
                    M[i][j] = d[target_area][target_pop][source_area][source_pop]
                    j += 1
            M[i][j] = d[target_area][target_pop]['external']['external']
            i += 1
    return M


def matrix_to_dict(m, area_list, structure, external=None):
    """
    Convert a matrix containing connectivity
    information of a network defined by structure to a dictionary.
    Parameters
    ----------
    m : array-like
        Matrix to be converted.
    area_list: list
        List of areas in the network. Defines the order of areas
        in the matrix to be created.
    structure : dict
        Structure of the network. Define the populations for each single area.
    external: numpy.ndarray or dict
        If None, do not include connectivity from external
        sources in the return dictionary.
        If numpy.ndarray or dict, use the connectivity given to add an entry
        'external' for each population.
        Defaults to None.
    """
    dic = nested_dict()
    for area, area2 in product(area_list, area_list):
        mask = create_mask(
            structure, target_areas=[area], source_areas=[area2], external=False)
        if external is not None:
            x = m[mask[:, :-1]]
        else:
            x = m[mask]

        if area == 'TH' and area2 == 'TH':
            x = x.reshape((6, 6))
            x = np.insert(x, 2, np.zeros((2, 6), dtype=float), axis=0)
            x = np.insert(x, 2, np.zeros((2, 8), dtype=float), axis=1)
        elif area2 == 'TH':
            x = x.reshape((8, 6))
            x = np.insert(x, 2, np.zeros((2, 8), dtype=float), axis=1)
        elif area == 'TH':
            x = x.reshape((6, 8))
            x = np.insert(x, 2, np.zeros((2, 8), dtype=float), axis=0)
        else:
            x = x.reshape((8, 8))
        for i, pop in enumerate(population_list):
            for j, pop2 in enumerate(population_list):
                if np.isclose(0., x[i][j]):
                    x[i][j] = 0.
                dic[area][pop][area2][pop2] = x[i][j]
    if external is not None:
        if isinstance(external, np.ndarray):
            for area in dic:
                for pop in population_list:
                    if pop in structure[area]:
                        mask = create_vector_mask(
                            structure, areas=[area], pops=[pop])
                        dic[area][pop]['external'] = {
                            'external': external[mask][0]}
                    else:
                        dic[area][pop]['external'] = {
                            'external': 0.}

        if isinstance(external, dict):
            for area in dic:
                for pop in dic[area]:
                    dic[area][pop]['external'] = external[
                        area][pop]

    return dic.to_dict()


def vector_to_dict(v, area_list, structure, external=None):
    """
    Convert a vector containing neuron numbers
    of a network defined by structure to a dictionary.
    Parameters
    ----------
    v : array-like
        Vector to be converted.
    area_list: list
        List of areas in the network. Defines the order of areas
        in the vector to be created.
    structure : dict
        Structure of the network. Define the populations for each single area.
    """
    dic = nested_dict()
    for area in area_list:
        vmask = create_vector_mask(structure, areas=[area])
        for i, pop in enumerate(structure[area]):
            dic[area][pop] = v[vmask][i]
        for pop in population_list:
            if pop not in structure[area]:
                dic[area][pop] = 0.

        dic[area]['total'] = sum(v[vmask])
    return dic.to_dict()


def dict_to_vector(d, area_list, structure):
    """
    Convert a dictionary containing population sizes
    of a network defined by structure to a vector.
    Parameters
    ----------
    d : dict
        Dictionary to be converted.
    area_list: list
        List of areas in the network. Defines the order of areas
        in the given vector.
    structure : dict
        Structure of the network. Define the populations for each single area.
    """
    dim = 0
    for area in structure.keys():
        dim += len(structure[area])

    V = np.zeros(dim)
    i = 0
    for target_area in area_list:
        if target_area in structure:
            for target_pop in structure[target_area]:
                if isinstance(d[target_area][target_pop], Iterable):
                    V[i] = d[target_area][target_pop][0]
                else:
                    V[i] = d[target_area][target_pop]
                i += 1
    return V


def convert_syn_weight(W, neuron_params):
    """
    Convert the amplitude of the PSC into mV.
    Parameters
    ----------
    W : float
        Synaptic weight defined as the amplitude of the post-synaptic current.
    neuron_params : dict
        Parameters of the neuron.
    """
    tau_syn_ex = neuron_params['tau_syn_ex']
    C_m = neuron_params['C_m']
    PSP_transform = tau_syn_ex / C_m

    return PSP_transform * W

def scale_network(self):
        """
        Scale the network if `N_scaling` and/or `K_scaling` differ from 1.
        This function:
        - adjusts the synaptic weights such that the population-averaged
          stationary spike rates approximately match the given `full-scale_rates`.
        - scales the population sizes with `N_scaling` and indegrees with `K_scaling`.
        - scales the synapse numbers with `N_scaling`*`K_scaling`.
        """
        # population sizes
        self.N_vec *= self.params['N_scaling']

        # Scale the synaptic weights before the indegrees to use full-scale indegrees
        self.adj_W_to_K()
        # Then scale the indegrees and synapse numbers
        self.K_matrix *= self.params['K_scaling']
        self.syn_matrix *= self.params['K_scaling'] * self.params['N_scaling']

        # Finally recreate dictionaries
        self.N = vector_to_dict(self.N_vec, self.area_list, self.structure)
        self.K = matrix_to_dict(self.K_matrix[:, :-1], self.area_list,
                                self.structure, external=self.K_matrix[:, -1])
        self.W = matrix_to_dict(self.W_matrix[:, :-1], self.area_list,
                                self.structure, external=self.W_matrix[:, -1])

        self.synapses = matrix_to_dict(self.syn_matrix, self.area_list, self.structure)

def vectorize(self):
    """
    Create matrix and vector version of neuron numbers, synapses
    and synapse weight dictionaries.
    """

    self.N_vec = dict_to_vector(self.N, self.area_list, self.structure)
    self.syn_matrix = dict_to_matrix(self.synapses, self.area_list, self.structure)
    self.K_matrix = dict_to_matrix(self.K, self.area_list, self.structure)
    self.W_matrix = dict_to_matrix(self.W, self.area_list, self.structure)
    self.J_matrix = convert_syn_weight(self.W_matrix,
                                       self.params['neuron_params']['single_neuron_dict'])
    self.structure_vec = ['-'.join((area, pop)) for area in
                          self.area_list for pop in self.structure[area]]
    self.add_DC_drive = np.zeros_like(self.N_vec)

def adj_W_to_K(self):
    """
    Adjust weights to scaling of neuron numbers and indegrees.
    The recurrent and external weights are adjusted to the scaling
    of the indegrees. Extra DC input is added to compensate the scaling
    and preserve the mean and variance of the input.
    """
    tau_m = self.params['neuron_params']['single_neuron_dict']['tau_m']
    C_m = self.params['neuron_params']['single_neuron_dict']['C_m']

    if isinstance(self.params['fullscale_rates'], np.ndarray):
        raise ValueError("Not supported. Please store the "
                         "rates in a file and define the path to the file as "
                         "the parameter value.")
    else:
        with open(self.params['fullscale_rates'], 'r') as f:
            d = json.load(f)
        full_mean_rates = dict_to_vector(d, self.area_list, self.structure)

    rate_ext = self.params['input_params']['rate_ext']
    J_ext = self.J_matrix[:, -1]
    K_ext = self.K_matrix[:, -1]
    x1_ext = 1e-3 * tau_m * J_ext * K_ext * rate_ext
    x1 = 1e-3 * tau_m * np.dot(self.J_matrix[:, :-1] * self.K_matrix[:, :-1], full_mean_rates)
    K_scaling = self.params['K_scaling']
    self.J_matrix /= np.sqrt(K_scaling)
    self.add_DC_drive = C_m / tau_m * ((1. - np.sqrt(K_scaling)) * (x1 + x1_ext))
    neuron_params = self.params['neuron_params']['single_neuron_dict']
    self.W_matrix = (1. / convert_syn_weight(1., neuron_params) * self.J_matrix)