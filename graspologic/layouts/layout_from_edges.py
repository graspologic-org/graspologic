# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import time
import math
import logging

logger = logging.getLogger(__name__)
from .layouts import NodePosition
import networkx
from typing import List, Optional


color_file_json = "colors-100.json"


def _partition_graph(graph: networkx.Graph):
    import graspologic_native as gln

    # find our own communities
    logger.info(
        f"nodes and edges in network: {graph.number_of_nodes()}, {graph.number_of_edges()}"
    )
    edges = [(s, t, float(w)) for s, t, w in graph.edges(data="weight")]
    logger.info(f"edges in network: {len(edges)}")
    clustering_improved, modularity, partition = gln.leiden(edges)
    return partition


def _creat_node2vec_embedding(graph: networkx.Graph):
    """Creates a node2vec embedding of the graph with a set of hard coded
    defaults to the embedding algorithm

    Returns: an arary of numpy.Array and a list of labels
    """
    import graspologic as gspl

    start = time.time()
    tensors, labels = gspl.embed.node2vec_embed(
        graph=graph, dimensions=128, num_walks=10, window_size=2, iterations=3
    )
    embedding_time = time.time() - start
    logger.info(f"embedding completed in {embedding_time} seconds")
    return tensors, labels


def _get_node_colors(
    color_file,
    partition,
    light_background=True,
    color_is_continuous=False,
    color_scheme="nominal",
    node_attributes=None,
):
    from ._helpers import (
        get_node_colors_from_partition,
        create_colormap,
        read_json_colorfile,
        get_partition,
    )

    light_colors, dark_colors = read_json_colorfile(color_file)
    if light_background:
        color_list = light_colors[color_scheme]
    else:
        color_list = dark_colors[color_scheme]

    if color_is_continuous:
        if node_attributes is None:
            raise Exception(
                "Need node_file, node_id, and color_attribute to specify continous value"
            )
        node_colors = get_sequential_node_colors(color_list, node_attributes)
    else:
        partition = get_partition(partition, node_attributes)
        colormap = create_colormap(color_list, partition)
        node_colors = get_node_colors_from_partition(partition, colormap)

    return node_colors


def layout_with_node2vec_umap(
    graph: networkx.Graph,
    min_dist: float = 0.75,
    n_neighbors: int = 25,
    max_edges=10000000,
):
    """
    Performs edge cuts to ge the number of edges under 1M edges.
    Finds the largeest connected component, performs a node2vec embedding on the LCC.
    Runs UMAP to transform to 2D, then removes node overlaps.
    @return the LCC and the node positions
    """
    import umap

    lcc = _cut_edges_if_needed(graph, max_edges_to_keep=max_edges)
    tensors, labels = _creat_node2vec_embedding(lcc)
    logger.debug(f"tensors: {len(tensors)}, labels: {len(labels)}")

    start = time.time()
    neighbors = min(n_neighbors, len(tensors) - 1)
    transformed_points = umap.UMAP(
        min_dist=min_dist, n_neighbors=neighbors
    ).fit_transform(tensors)
    umap_time = time.time() - start
    logger.info(f"umap completed in {umap_time} seconds")

    nx_graph, positions = _make_graph_and_positions(
        labels, transformed_points, lcc.degree()
    )
    non_overlapping_positions = remove_overlaps(positions)

    return nx_graph, non_overlapping_positions


def layout_with_node2vec_tsne(
    graph: networkx.Graph,
    perplexity: int,
    n_iters: int,
    max_edges_to_keep: int = 10000000,
):
    """
    Performs edge cuts to ge the number of edges under 1M edges.
    Finds the largeest connected component, performs a node2vec embedding on the LCC.
    Runs t-SNE to transform to 2D, then removes node overlaps.
    @return the LCC as a networkx.Graph and the node positions as a list of NodePosition objects
    """
    from sklearn.manifold import TSNE

    lcc = _cut_edges_if_needed(graph, max_edges_to_keep=10000000)
    tensors, labels = _creat_node2vec_embedding(lcc)

    start = time.time()
    transformed_points = TSNE(perplexity=perplexity, n_iter=n_iters).fit_transform(
        tensors
    )
    tsne_time = time.time() - start
    logger.info(
        f"tsne completed in {tsne_time} seconds with peplexity: {perplexity} and n_iters: {n_iters}"
    )

    nx_graph, positions = _make_graph_and_positions(
        labels, transformed_points, lcc.degree()
    )
    non_overlapping_positions = remove_overlaps(positions)

    return nx_graph, non_overlapping_positions


def layout_node2vec_umap_from_file(
    edge_file: str,
    image_file: Optional[str],
    location_file: Optional[str],
    dpi: int,
    max_edges: int = 10000000,
):
    """
    Reads a graph from a weighted edgelist file. Performs edge cuts if needed.
    Computes a node2vec embedding.
    Then performs a UMAP dimensionality reduction.
    """
    from .layouts import save_graph
    from ._helpers import read_graph

    graph = read_graph(edge_file)
    lcc, positions = layout_with_node2vec_umap(graph, max_edges=max_edges)
    partition = _partition_graph(graph)

    node_colors = _get_node_colors(
        color_file_json,
        partition,
        light_background=True,
        color_is_continuous=False,
        color_scheme="nominal",
        node_attributes=None,
    )

    if location_file is not None:
        write_locations(location_file, positions, partition)
    if image_file is not None:
        save_graph(
            image_file,
            lcc,
            positions,
            node_colors=node_colors,
            dpi=dpi,
            light_background=True,
        )
    return


def layout_node2vec_tsne_from_file(
    edge_file: str,
    image_file: Optional[str],
    location_file: Optional[str],
    dpi: int,
    perplexity: int = 30,
    n_iters: int = 1000,
    max_edges: int = 10000000,
):
    """
    Reads a graph from a weighted edgelist file. Performs edge cuts if needed.
    Computes a node2vec embedding.
    Then performs a t-SNE dimensionality reduction.
    """
    from ._helpers import read_graph
    from .layouts import save_graph

    graph = read_graph(edge_file)
    print(
        f"nodes, edges after read: {graph.number_of_nodes()}, {graph.number_of_edges()}"
    )
    lcc, positions = layout_with_node2vec_tsne(
        graph, perplexity, n_iters, max_edges_to_keep=max_edges
    )
    partition = _partition_graph(graph)

    node_colors = _get_node_colors(
        color_file_json,
        partition,
        light_background=True,
        color_is_continuous=False,
        color_scheme="nominal",
        node_attributes=None,
    )

    if location_file is not None:
        write_locations(location_file, positions, partition)
    if image_file is not None:
        save_graph(
            image_file,
            lcc,
            positions,
            node_colors=node_colors,
            dpi=dpi,
            light_background=True,
        )

    return


def remove_overlaps(list_of_node_positions: List[NodePosition]):
    from .nooverlap import quad_tree, node

    logger.info("removing overlaps")
    local_nodes = []
    for n in list_of_node_positions:
        local_nodes.append(node(n.node_id, n.x, n.y, n.size, n.community))
    qt = quad_tree(local_nodes, 50)
    start = time.time()
    # qt.layout_dense_first(first_color='#FF0004')
    qt.layout_dense_first(first_color=None)
    stop = time.time()
    logger.info(f"removed overlap in {stop-start} seconds")

    new_positions = []
    for n in local_nodes:
        new_positions.append(
            NodePosition(
                node_id=n.nid, x=n.x, y=n.y, size=n.size, community=n.community
            )
        )
    return new_positions


def _extract_positions_from_graph(graph, id_to_community):
    new_graph = networkx.Graph()
    positions = []
    for key, att in graph.nodes(data=True):
        data = att["attributes"][0]
        x = float(data["x"])
        y = float(data["y"])
        size = float(data["size"])

        new_graph.add_node(key)
        if key in id_to_community:
            community = (
                id_to_community[key] if type(id_to_community[key]) == int else None
            )
            positions.append(
                NodePosition(node_id=key, x=x, y=y, size=size, community=community)
            )
    return new_graph, positions


def _count_edges(graph, weight_ge, weight_att="weight"):
    count = 0
    for s, t, att in graph.edges(data=True):
        if att[weight_att] >= weight_ge:
            count += 1
    return count


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


def _cut_edges_if_needed(graph, max_edges_to_keep=1000000):
    from ._helpers import largest_connected_component
    from graspologic.preprocessing import cut_edges_by_weight, histogram_edge_weight

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


# this the the target density for the non-overlapping graph. This is based on
# many subjective experiments.
TARGET_RATIO = 0.14


def _new_bounds(minx, miny, maxx, maxy, covered_area):
    range_x = maxx - minx
    mid_x = minx + range_x / 2
    range_y = maxy - miny
    mid_y = miny + range_y / 2
    range_ratio = range_x / range_y
    # current_ratio = covered_area/area
    area = range_x * range_y
    new_area = covered_area / TARGET_RATIO
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


def _scale_points(points, covered_area):
    moved_points = _center_points_on_origin(points)
    minx, miny, maxx, maxy = _get_bounds(moved_points)
    # print ("centered bounds", minx, miny, maxx, maxy)
    new_minx, new_miny, new_maxx, new_maxy = _new_bounds(
        minx, miny, maxx, maxy, covered_area
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


def _make_graph_and_positions(vertex_labels, points, degrees):
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


def write_locations(filename, positions, partition):
    import csv

    with open(filename, "w") as ofile:
        writer = csv.writer(ofile)
        writer.writerow(["ID", "x", "y", "size", "community"])
        for node in positions:
            writer.writerow(
                [node.node_id, node.x, node.y, node.size, partition[node.node_id]]
            )


if __name__ == "__main__":
    _main()
