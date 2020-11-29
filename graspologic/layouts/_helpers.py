# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import atexit
import csv
from collections import defaultdict
import json
import logging
import math
import networkx
import numpy
import os
from pathlib import Path
import pkg_resources
from sklearn.preprocessing import minmax_scale
import time
from typing import Any, Callable, Tuple, Union, Generator, Set, List


import graspologic
from graspologic.layouts import NodePosition
from graspologic.preprocessing import cut_edges_by_weight, histogram_edge_weight

logger = logging.getLogger(__name__)


def ensure_directory_for_file(filename: str):
    """
    Assumes filename is an output filename. Finds the directory that filename exists in
    and creates it and parent directories if they don't exists.
    """
    if filename is None:
        return
    head, tail = os.path.split(filename)
    if head != "":
        os.makedirs(head, exist_ok=True)
    return


def read_node_file(filename, id_atttribute, color_attribute):
    if filename is None:
        return None
    if id_atttribute is None or color_attribute is None:
        raise ValueError("Must specify a color_attribute if a node_file is specified")
    start = time.time()
    partition = {}
    with open(filename, "r") as ifile:
        reader = csv.DictReader(ifile)
        for row in reader:
            partition[row[id_atttribute]] = row[color_attribute]
    read_time = time.time() - start
    logger.info(f"read node file in : {read_time} seconds")
    return partition


def get_node_colors_from_partition(partition, colormap):
    return {n: colormap[c] for n, c in partition.items()}


def get_sequential_node_colors(color_list, node_attributes, use_log_scale):
    num_colors = len(color_list)
    logger.debug(f"num colors: {num_colors}")
    keys = []
    values = []
    for nid, svalue in node_attributes.items():
        keys.append(nid)
        values.append(float(svalue))

    if use_log_scale:
        min_value = min(values)
        mmax_value = max(values)
        values = [math.log(fvalue) for fvalue in values]
        # values.append(math.log(fvalue))

    np_values = numpy.array(values).reshape(1, -1)
    new_values = minmax_scale(np_values, feature_range=(0, num_colors - 1), axis=1)
    logger.debug(f"len(values): {len(values)}, {len(node_attributes)}")
    logger.debug(f"before min: {np_values.min()}, max: {np_values.max()}")
    logger.debug(f"after  min: {new_values.min()}, max: {new_values.max()}")
    node_colors = {}
    for idx, nid in enumerate(keys):
        index = int(new_values[0, idx])
        # index = min(index, len(color_list)-1)
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
    for community in sorted(
        community_size, reverse=True, key=lambda x: community_size[x]
    ):
        colormap[community] = color_list[next_comm % color_list_size]
        next_comm += 1
    return colormap


def read_json_colorfile(filename):
    if Path(filename).is_file():
        colors_path = filename
    else:
        atexit.register(pkg_resources.cleanup_resources)
        include_path = pkg_resources.resource_filename(__package__, "include")
        colors_path = os.path.join(include_path, filename)

    with open(colors_path) as ifile:
        jobj = json.load(ifile)
    light = jobj["light"]
    dark = jobj["dark"]
    return light, dark


def get_partition(partitions, node_attributes):
    if node_attributes is None:
        return partitions
    return node_attributes


def read_locations(filename):
    logger.info(f"reading {filename}")
    with open(filename, "r") as ifile:
        reader = csv.DictReader(
            ifile,
        )
        # ["ID", "x", "y", "size", "community"]
        node_positions = []
        partition = {}
        for row in reader:
            node_id = row["ID"]
            partition[node_id] = row["community"]
            node_positions.append(
                NodePosition(
                    node_id=node_id,
                    x=float(row["x"]),
                    y=float(row["y"]),
                    size=float(row["size"]),
                    community=None,
                )
            )
        return node_positions, partition


def read_graph(edge_file, has_header=True):
    start = time.time()
    with open(edge_file, "r") as ifile:
        graph = networkx.Graph()
        reader = csv.reader(ifile)
        if has_header:
            # Need to read the header
            next(reader)
        for row in reader:
            source = row[0]
            target = row[1]
            weight = float(row[2])
            graph.add_edge(source, target, weight=weight)
    read_time = time.time() - start
    logger.info(f"read edge list file in : {read_time} seconds")
    return graph


def largest_connected_component(
    graph: networkx.Graph, weakly: bool = True
) -> networkx.Graph:
    """
    Returns the largest connected component of the graph.

    :param networkx.Graph graph: The networkx graph object to select the largest connected component from.
      Can be either directed or undirected.
    :param bool weakly: Whether to find weakly connected components or strongly connected components for directed
      graphs.
    :return: A copy of the largest connected component as an networkx.Graph object
    :rtype: networkx.Graph
    """
    connected_component_function = _connected_component_func(graph, weakly)
    largest_component = max(connected_component_function(graph), key=len)
    return graph.subgraph(largest_component).copy()


def _connected_component_func(
    graph: networkx.Graph, weakly: bool = True
) -> Callable[[networkx.Graph], Generator[Set[Any], None, None]]:
    if not isinstance(graph, networkx.Graph):
        raise TypeError("graph must be a networkx.Graph")
    if not networkx.is_directed(graph):
        return networkx.connected_components
    elif weakly:
        return networkx.weakly_connected_components
    else:
        return networkx.strongly_connected_components


# TODO: there is a better way to do this than the brute force way, but it is
# such a small percentage of the total time that I have not come back to fix it
# yet.
def _find_right_edge_count(graph, max_edges, bins):
    best_weight_cut = bins[0] + 1
    logger.debug(f"checking: best_weight_cut {best_weight_cut}")
    ecount = _count_edges(graph, best_weight_cut)
    # there is a more efficient way to do this for sure
    while ecount > max_edges:
        best_weight_cut += 1
        logger.debug(f"checking: best_weight_cut {best_weight_cut}, ecount: {ecount}")
        ecount = _count_edges(graph, best_weight_cut)
    logger.debug(f"cut using best_weight_cut {best_weight_cut}, ecount: {ecount}")
    return best_weight_cut


def cut_edges_if_needed(graph, max_edges_to_keep=1000000):

    num_edges = len(graph.edges())
    num_nodes = len(graph.nodes())

    logger.info(f"num edges: {num_edges}")
    logger.info(f"num node: {num_nodes}")
    if num_edges > max_edges_to_keep:
        histogram, bins = histogram_edge_weight(graph)
        # print (histogram)
        edge_cut_weight = _find_right_edge_count(graph, max_edges_to_keep, bins)
        new_graph = cut_edges_by_weight(
            graph, cut_threshold=edge_cut_weight, cut_process="smaller_than_inclusive"
        )
        logger.debug(f"after cut num edges: {len(new_graph.edges())}")
        logger.debug(f"after cut num node: {len(new_graph.nodes())}")
        lcc = largest_connected_component(new_graph)
    else:
        lcc = largest_connected_component(graph)
    len_lcc = len(lcc.edges())
    logger.info(f"num edges in lcc: {len_lcc}")
    logger.info(f"num nodes in lcc: {len(lcc.nodes())}")
    return lcc


def _find_min_max_degree(degrees):
    min_degree = math.inf
    max_degree = -math.inf
    for i, d in degrees:
        min_degree = min(min_degree, d)
        max_degree = max(max_degree, d)
    return min_degree, max_degree


def _compute_sizes(degrees, min_degree, max_degree, min_size=5.0, max_size=150.0):
    sizes = {}
    for i, d in degrees:
        if max_degree == min_degree:
            size = min_size
        else:
            normalized = (d - min_degree) / (max_degree - min_degree)
            size = normalized * (max_size - min_size) + min_size
        sizes[i] = size
    return sizes


def _covered_size(sizes):
    total = 0.0
    for k, v in sizes.items():
        total += v * v * math.pi
    return total


def _get_bounds(points):
    minx, miny, maxx, maxy = math.inf, math.inf, -math.inf, -math.inf
    for x, y in points:
        minx = min(x, minx)
        miny = min(y, miny)
        maxx = max(x, maxx)
        maxy = max(y, maxy)
    return minx, miny, maxx, maxy


def _center_points_on_origin(points):
    minx, miny, maxx, maxy = _get_bounds(points)
    # print (minx, miny, maxx, maxy)
    x_range = maxx - minx
    y_range = maxy - miny
    center_x = x_range / 2 + minx
    center_y = y_range / 2 + miny
    moved_points = []
    for x, y in points:
        moved_points.append([x - center_x, y - center_y])

    minx, miny, maxx, maxy = (
        minx - center_x,
        miny - center_y,
        maxx - center_x,
        maxy - center_y,
    )
    # print (minx, miny, maxx, maxy)
    return moved_points


def _scale_to_unit_square(points):
    minx, miny, maxx, maxy = _get_bounds(points)
    scaled_points = []
    for x, y in points:
        scaled_points.append([x / maxx, y / maxy])
    return scaled_points


def _scale_to_new_bounds(points, minx, miny, maxx, maxy):
    # x_range = maxx - minx
    # y_range = maxy - miny
    scaled = []
    for x, y in points:
        scaled.append([x * maxx, y * maxy])
    return scaled


def _new_bounds(minx, miny, maxx, maxy, covered_area, target_ratio):
    range_x = maxx - minx
    mid_x = minx + range_x / 2
    range_y = maxy - miny
    mid_y = miny + range_y / 2
    range_ratio = range_x / range_y
    # current_ratio = covered_area/area
    area = range_x * range_y
    new_area = covered_area / target_ratio
    new_range_y = math.sqrt(new_area / range_ratio)
    new_range_x = new_area / new_range_y
    new_minx, new_miny, new_maxx, new_maxy = (
        mid_x - new_range_x / 2,
        mid_y - new_range_y / 2,
        mid_x + new_range_x / 2,
        mid_y + new_range_y / 2,
    )
    # print (minx, miny, maxx, maxy)
    # print (new_minx, new_miny, new_maxx, new_maxy)

    return new_minx, new_miny, new_maxx, new_maxy


## through lots of experiments we have found 14% to be a good ratio
## of color to whitespace
def _scale_points(points, covered_area, target_ratio=0.14):
    moved_points = _center_points_on_origin(points)
    minx, miny, maxx, maxy = _get_bounds(moved_points)
    # print ("centered bounds", minx, miny, maxx, maxy)
    new_minx, new_miny, new_maxx, new_maxy = _new_bounds(
        minx, miny, maxx, maxy, covered_area, target_ratio=target_ratio
    )
    # print ("new bounds", new_minx, new_miny, new_maxx, new_maxy)
    scaled_points = _scale_to_unit_square(moved_points)
    minx, miny, maxx, maxy = _get_bounds(scaled_points)
    # print ("unit square", minx, miny, maxx, maxy)
    final_locs = _scale_to_new_bounds(
        scaled_points, new_minx, new_miny, new_maxx, new_maxy
    )
    # minx, miny, maxx, maxy = _get_bounds(final_locs)
    # print ("final locs", minx, miny, maxx, maxy)

    return final_locs


def create_node2vec_embedding(graph: networkx.Graph):
    """Creates a node2vec embedding of the graph with a set of hard coded
    defaults to the embedding algorithm

    Returns: an arary of numpy.array and a list of labels
    """

    start = time.time()
    tensors, labels = graspologic.embed.node2vec_embed(
        graph=graph, dimensions=128, num_walks=10, window_size=2, iterations=3
    )
    embedding_time = time.time() - start
    logger.info(f"embedding completed in {embedding_time} seconds")
    return tensors, labels


def make_graph_and_positions(vertex_labels, points, degrees):
    min_degree, max_degree = _find_min_max_degree(degrees)
    sizes = _compute_sizes(degrees, min_degree, max_degree)
    covered_area = _covered_size(sizes)
    scaled_points = _scale_points(points, covered_area)
    graph = networkx.Graph()
    positions = []
    for idx, key in enumerate(vertex_labels):
        graph.add_node(key)
        positions.append(
            NodePosition(
                node_id=key,
                x=scaled_points[idx][0],
                y=scaled_points[idx][1],
                size=sizes[key],
                community=None,
            )
        )
    return graph, positions
