# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import csv
import logging
import networkx
import time
from typing import List, Optional
import umap


from graspologic.layouts import NodePosition, save_graph
import graspologic_native as gln
from graspologic.layouts._helpers import (
    create_colormap,
    create_node2vec_embedding,
    cut_edges_if_needed,
    get_node_colors_from_partition,
    get_partition,
    largest_connected_component,
    make_graph_and_positions,
    read_graph,
    read_json_colorfile,
)
from graspologic.layouts.nooverlap import remove_overlaps
import graspologic.layouts.tsne


logger = logging.getLogger(__name__)

color_file_json = "colors-100.json"


def _partition_graph(graph: networkx.Graph):

    # find our own communities
    logger.info(
        f"nodes and edges in network: {graph.number_of_nodes()}, {graph.number_of_edges()}"
    )
    edges = [(s, t, float(w)) for s, t, w in graph.edges(data="weight")]
    logger.info(f"edges in network: {len(edges)}")
    clustering_improved, modularity, partition = gln.leiden(edges)
    return partition


def _get_node_colors(
    color_file,
    partition,
    light_background=True,
    color_is_continuous=False,
    color_scheme="nominal",
    node_attributes=None,
):
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

    lcc = cut_edges_if_needed(graph, max_edges_to_keep=max_edges)
    tensors, labels = create_node2vec_embedding(lcc)
    logger.debug(f"tensors: {len(tensors)}, labels: {len(labels)}")

    start = time.time()
    neighbors = min(n_neighbors, len(tensors) - 1)
    transformed_points = umap.UMAP(
        min_dist=min_dist, n_neighbors=neighbors
    ).fit_transform(tensors)
    umap_time = time.time() - start
    logger.info(f"umap completed in {umap_time} seconds")

    nx_graph, positions = make_graph_and_positions(
        labels, transformed_points, lcc.degree()
    )
    non_overlapping_positions = remove_overlaps(positions)

    return nx_graph, non_overlapping_positions


def layout_node2vec_umap_from_file(
    edge_file: str,
    image_file: Optional[str],
    location_file: Optional[str],
    dpi: int,
    skip_header: bool = False,
    max_edges: int = 10000000,
):
    """
    Reads a graph from a weighted edgelist file. Performs edge cuts if needed.
    Computes a node2vec embedding.
    Then performs a UMAP dimensionality reduction.
    """

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

    lcc = cut_edges_if_needed(graph, max_edges_to_keep=max_edges_to_keep)
    tensors, labels = create_node2vec_embedding(lcc)

    transformed_points = graspologic.layouts.tsne.reduce_dimensions(
        tensors, perplexity, n_iters
    )

    nx_graph, positions = make_graph_and_positions(
        labels, transformed_points, lcc.degree()
    )
    non_overlapping_positions = remove_overlaps(positions)

    return nx_graph, non_overlapping_positions


def layout_node2vec_tsne_from_file(
    edge_file: str,
    image_file: Optional[str],
    location_file: Optional[str],
    dpi: int,
    skip_header: bool = False,
    perplexity: int = 30,
    n_iters: int = 1000,
    max_edges: int = 10000000,
):
    """
    Reads a graph from a weighted edgelist file. Performs edge cuts if needed.
    Computes a node2vec embedding.
    Then performs a t-SNE dimensionality reduction.
    """

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


def write_locations(filename, positions, partition):
    with open(filename, "w") as ofile:
        writer = csv.writer(ofile)
        writer.writerow(["ID", "x", "y", "size", "community"])
        for node in positions:
            writer.writerow(
                [node.node_id, node.x, node.y, node.size, partition[node.node_id]]
            )


if __name__ == "__main__":
    _main()
