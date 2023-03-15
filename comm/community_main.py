# -*- coding: utf-8 -*-
"""
This module implements community detection.
"""
from __future__ import print_function

import array
import math

import numbers
from random import random
import warnings
import random
import networkx as nx
import numpy as np
from numpy.random import laplace
import time

from .community_status import Status



#__PASS_MAX = -1
__PASS_MAX = 10000
__MIN = 0.0000001


def check_random_state(seed):
    
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, (numbers.Integral, np.integer)):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError("%r cannot be used to seed a numpy.random.RandomState"
                     " instance" % seed)


def partition_at_level(dendrogram, level):
    
    partition = dendrogram[0].copy()
    for index in range(1, level + 1):
        for node, community in partition.items():
            partition[node] = dendrogram[index][community]
    return partition


def modularity(partition, graph, weight='weight'):
   
    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    inc = dict([])
    deg = dict([])
    # links stands for the number of edge
    links = graph.size(weight=weight)
    if links == 0:
        raise ValueError("A graph without link has an undefined modularity")

    for node in graph:
        # com stands for the node's corresponding community
        com = partition[node]
        # deg[com] is used to storage the degree of relative community
        deg[com] = deg.get(com, 0.) + graph.degree(node, weight=weight)
        for neighbor, datas in graph[node].items():
            # if the result of get() is None, return 1
            edge_weight = datas.get(weight, 1)
            if partition[neighbor] == com:
                if neighbor == node:
                    inc[com] = inc.get(com, 0.) + float(edge_weight)
                else:
                    inc[com] = inc.get(com, 0.) + float(edge_weight) / 2.

    res = 0.
    for com in set(partition.values()):
        # calculate the modularity based on the formula: Q = deg_in/m - (deg_com/(2m))^2
        res += (inc.get(com, 0.) / links) - \
               (deg.get(com, 0.) / (2. * links)) ** 2
    return res


def best_partition(graph,
                   partition=None,
                   weight='weight',
                   resolution=1.,
                   randomize=None,
                   random_state=None,
                   epsilon_EM=None,
                   divide=1):

    dendo = generate_dendrogram(graph,
                                partition,
                                weight,
                                resolution,
                                randomize,
                                random_state,
                                epsilon_EM,
                                divide)
    return partition_at_level(dendo, len(dendo) - 1)

def generate_dendrogram(graph,
                        part_init=None,
                        weight='weight',
                        resolution=1.,
                        randomize=None,
                        random_state=None,
                        epsilon_EM=None,
                        divide=1):

    if graph.is_directed():
        raise TypeError("Bad graph type, use only non directed graph")

    # Properly handle random state, eventually remove old `randomize` parameter
    # NOTE: when `randomize` is removed, delete code up to random_state = ...
    if randomize is not None:
        warnings.warn("The `randomize` parameter will be deprecated in future "
                      "versions. Use `random_state` instead.", DeprecationWarning)
        # If shouldn't randomize, we set a fixed seed to get determinisitc results
        if randomize is False:
            random_state = 0

    # We don't know what to do if both `randomize` and `random_state` are defined
    if randomize and random_state is not None:
        raise ValueError("`randomize` and `random_state` cannot be used at the "
                         "same time")

    random_state = check_random_state(random_state)

    # special case, when there is no link
    # the best partition is everyone in its community
    if graph.number_of_edges() == 0:
        part = dict([])
        for i, node in enumerate(graph.nodes()):
            part[node] = i
        return [part]

    current_graph = graph.copy()




    status = Status()
    status.init(current_graph, weight, part_init)
    # status.init(current_graph, weight, part1)
    status_list = list()


    v1 = np.sum(list(status.internals.values()))
    # print('initial internals:%d'%v1)

    t1 = time.time()
    __comm_adjust_em(current_graph, status, weight, resolution, random_state, epsilon_EM  , divide)

    v1 = np.sum(list(status.internals.values()))
    # print('final internals:%d'%v1)

    # print('adjust time:%.2fs'%(time.time()-t1))
    new_mod = __modularity(status, resolution)
    partition = __renumber(status.node2com)
    status_list.append(partition)
    mod = new_mod
    # induced_graph is to create new graph based on the partition
    current_graph = induced_graph(partition, current_graph, weight)
    status.init(current_graph, weight)

    return status_list[:]


def induced_graph(partition, graph, weight="weight"):

    ret = nx.Graph()
    ret.add_nodes_from(partition.values())

    for node1, node2, datas in graph.edges(data=True):
        edge_weight = datas.get(weight, 1)
        com1 = partition[node1]
        com2 = partition[node2]
        w_prec = ret.get_edge_data(com1, com2, {weight: 0}).get(weight, 1)
        ret.add_edge(com1, com2, **{weight: w_prec + edge_weight})

    return ret


def __renumber(dictionary):

    values = set(dictionary.values())
    target = set(range(len(values)))

    if values == target:
        # no renumbering necessary
        ret = dictionary.copy()
    else:
        # add the values that won't be renumbered
        renumbering = dict(zip(target.intersection(values),
                               target.intersection(values)))
        # add the values that will be renumbered
        renumbering.update(dict(zip(values.difference(target),
                                    target.difference(values))))
        ret = {k: renumbering[v] for k, v in dictionary.items()}

    return ret


def load_binary(data):
    data = open(data, "rb")

    reader = array.array("I")
    reader.fromfile(data, 1)
    num_nodes = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_nodes)
    cum_deg = reader.tolist()
    num_links = reader.pop()
    reader = array.array("I")
    reader.fromfile(data, num_links)
    links = reader.tolist()
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    prec_deg = 0

    for index in range(num_nodes):
        last_deg = cum_deg[index]
        neighbors = links[prec_deg:last_deg]
        graph.add_edges_from([(index, int(neigh)) for neigh in neighbors])
        prec_deg = last_deg

    return graph

def __comm_adjust_em(graph, status, weight_key, resolution, random_state, epsilon, divide):
    
    nb_pass_done = 0
    cur_mod = __modularity(status, resolution)
    new_mod = cur_mod
  
    pass_max = round(divide)

    deltau = 1
    c1 = epsilon / (2 * pass_max * deltau * 2 )
    
  
    # print('epsilon:',c1)

    while nb_pass_done < pass_max:
        cur_mod = new_mod
        
        nb_pass_done += 1
        
        # iteration over the nodes
        for node in __randomize(graph.nodes(), random_state):
            
            com_node = status.node2com[node]
            
            # obtain all communities
            candi_communities = __allcom(node, graph, status, weight_key)
            
            remove_cost = - resolution * candi_communities.get(com_node,0)

            # remove the node from the original community
            __remove(node, com_node,
                    candi_communities.get(com_node, 0.), status)
            best_com = com_node
           

            coms = []
            incrs = []
            for com, dnc in __randomize(candi_communities.items(), random_state):
                incr = remove_cost + resolution * dnc
                incrs.append(incr)
                coms.append(com)
    
            incrs = np.array(incrs)
            incrs = incrs * c1
            incrs_m = max(np.max(incrs),0)
            exp_inc = np.exp(incrs-incrs_m)

            # Exponential Mechanism
            prob_inc = exp_inc / np.sum(exp_inc)
            best_com = np.random.choice(coms,p=prob_inc)
                
            # put the node into the best_com
            __insert(node, best_com,
                    candi_communities.get(best_com, 0.), status)
            
        new_mod = __modularity(status, resolution)
        


def __neighcom(node, graph, status, weight_key):

    weights = {}
    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            weights[neighborcom] = weights.get(neighborcom, 0) + edge_weight

    return weights



def __allcom(node, graph, status, weight_key):
    all_coms = list(status.node2com.values())
    candi_weights = dict.fromkeys(all_coms,0)

    for neighbor, datas in graph[node].items():
        if neighbor != node:
            edge_weight = datas.get(weight_key, 1)
            neighborcom = status.node2com[neighbor]
            candi_weights[neighborcom] = candi_weights.get(neighborcom, 0) + edge_weight

    return candi_weights 


def __remove(node, com, weight, status):
    status.degrees[com] = (status.degrees.get(com, 0.)
                           - status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) -
                                  weight - status.loops.get(node, 0.))
    status.node2com[node] = -1


def __insert(node, com, weight, status):

    status.node2com[node] = com
    status.degrees[com] = (status.degrees.get(com, 0.) +
                           status.gdegrees.get(node, 0.))
    status.internals[com] = float(status.internals.get(com, 0.) +
                                  weight + status.loops.get(node, 0.))


def __modularity(status, resolution):

    links = float(status.total_weight)
    result = 0.
    for community in set(status.node2com.values()):
        in_degree = status.internals.get(community, 0.)
        degree = status.degrees.get(community, 0.)
        if links > 0:
            result += in_degree * resolution / links -  ((degree / (2. * links)) ** 2)
    return result


def __randomize(items, random_state):
    randomized_items = list(items)
    random_state.shuffle(randomized_items)
    return randomized_items
