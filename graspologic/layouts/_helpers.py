# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import json
import time
import csv
import os
import pkg_resources
import atexit
from .layouts import NodePosition
from collections import defaultdict
import logging
import networkx as nx
import numpy
from typing import Any, Callable, Tuple, Union, Generator, Set, List
from enum import Enum
logger = logging.getLogger(__name__)


def ensure_directory_for_file(filename: str):
    '''
        Assumes filename is an output filename. Finds the directory that filename exists in
        and creates it and parent directories if they don't exists.
    '''
    if filename is None:
        return
    head, tail = os.path.split(filename)
    if head != '':
        os.makedirs(head, exist_ok=True)
    return


def read_node_file(filename, id_atttribute, color_attribute):
    if filename is None:
        return None
    if id_atttribute is None or color_attribute is None:
        raise ValueError("Must specify a color_attribute if a node_file is specified")
    start = time.time()
    partition = {}
    with open(filename, 'r') as ifile:
        reader = csv.DictReader(ifile)
        for row in reader:
            partition[row[id_atttribute]] = row[color_attribute]
    read_time = time.time() - start
    logger.info (f"read node file in : {read_time} seconds")
    return partition


def get_node_colors_from_partition(partition, colormap):
    return { n: colormap[c] for n, c in partition.items() }

def get_sequential_node_colors(color_list, node_attributes, use_log_scale):
    from sklearn.preprocessing import minmax_scale
    import math
    num_colors = len(color_list)
    logger.debug (f"num colors: {num_colors}")
    keys = []
    values = []
    for nid, svalue in node_attributes.items():
        keys.append(nid)
        values.append(float(svalue))

    if use_log_scale:
        min_value = min(values)
        mmax_value = max(values)
        values = [math.log(fvalue) for fvalue in values]
        #values.append(math.log(fvalue))

    np_values = numpy.array(values).reshape(1,-1)
    new_values = minmax_scale(np_values, feature_range=(0, num_colors-1) ,axis=1)
    logger.debug (f"len(values): {len(values)}, {len(node_attributes)}")
    logger.debug (f"before min: {np_values.min()}, max: {np_values.max()}")
    logger.debug (f"after  min: {new_values.min()}, max: {new_values.max()}")
    node_colors = {}
    for idx, nid in enumerate(keys):
        index = int(new_values[0,idx])
        #index = min(index, len(color_list)-1)
        color = color_list[index]
        node_colors[nid] = color

    return node_colors


def create_colormap(color_list, id_community):
    community_size = defaultdict(int)
    for comm in id_community.values():
        community_size[comm] += 1
    colormap = {}
    next_comm = 0
    color_list_size = len(color_list)
    # we wrap around if there are more communities than colors
    for community in sorted(community_size, reverse=True, key=lambda x: community_size[x]):
        colormap[community] = color_list[next_comm % color_list_size ]
        next_comm += 1
    return colormap


def read_json_colorfile(filename):
    from pathlib import Path
    if Path(filename).is_file():
        colors_path = filename
    else:
        atexit.register(pkg_resources.cleanup_resources)
        include_path = pkg_resources.resource_filename(__package__, "include")
        colors_path = os.path.join(include_path, filename)

    with open(colors_path) as ifile:
        jobj = json.load(ifile)
    light = jobj['light']
    dark = jobj['dark']
    return light, dark


def get_partition(partitions, node_attributes):
    if node_attributes is None:
        return partitions
    return node_attributes


def read_locations(filename):
    logger.info(f"reading {filename}")
    with open (filename, 'r') as ifile:
        reader = csv.DictReader(ifile,)
        #["ID", "x", "y", "size", "community"]
        node_positions = []
        partition = {}
        for row in reader:
            node_id = row["ID"]
            partition[node_id] =row["community"]
            node_positions.append(NodePosition(node_id=node_id, x=float(row["x"]), y=float(row["y"]), size=float(row["size"]), community=None))
        return node_positions, partition

def read_graph(edge_file, has_header=True):
    start = time.time()
    with open(edge_file, 'r') as ifile:
        graph = nx.Graph()
        if has_header:
            reader = csv.reader(ifile)
        #Need to read the header
        next(reader)
        for row in reader:
            source = row[0]
            target = row[1]
            weight = float(row[2])
            graph.add_edge(source, target, weight=weight)
    read_time = time.time() - start
    logger.info(f"read edge list file in : {read_time} seconds")
    return graph

def largest_connected_component(graph: nx.Graph, weakly: bool = True) -> nx.Graph:
    """
    Returns the largest connected component of the graph.

    :param networkx.Graph graph: The networkx graph object to select the largest connected component from.
      Can be either directed or undirected.
    :param bool weakly: Whether to find weakly connected components or strongly connected components for directed
      graphs.
    :return: A copy of the largest connected component as an nx.Graph object
    :rtype: networkx.Graph
    """
    connected_component_function = _connected_component_func(graph, weakly)
    largest_component = max(connected_component_function(graph), key=len)
    return graph.subgraph(largest_component).copy()

def _connected_component_func(
        graph: nx.Graph,
        weakly: bool = True
) -> Callable[[nx.Graph], Generator[Set[Any], None, None]]:
    if not isinstance(graph, nx.Graph):
        raise TypeError('graph must be a networkx.Graph')
    if not nx.is_directed(graph):
        return nx.connected_components
    elif weakly:
        return nx.weakly_connected_components
    else:
        return nx.strongly_connected_components
