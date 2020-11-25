# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import networkx
import logging
from graspologic.layouts import layouts
from graspologic.layouts._helpers import (
    read_node_file,
    get_node_colors_from_partition,
    create_colormap,
    read_locations,
    get_partition,
    read_json_colorfile,
    read_graph,
    get_sequential_node_colors,
)


logger = logging.getLogger(__name__)


def render_graph_from_files(
    edge_list_file,
    location_file,
    image_file,
    node_file,
    node_id,
    color_attribute,
    color_file,
    color_scheme,
    use_log_scale,
    color_is_continuous,
    light_background,
    vertex_alpha,
    vertex_line_width,
    edge_line_width,
    edge_alpha,
    figure_width,
    figure_height,
    vertex_shape,
    arrows,
    dpi,
):
    node_positions, partitions = read_locations(location_file)
    node_attributes = read_node_file(node_file, node_id, color_attribute)

    light_colors, dark_colors = read_json_colorfile(color_file)
    if light_background:
        color_list = light_colors[color_scheme]
    else:
        color_list = dark_colors[color_scheme]

    if color_is_continuous:
        if node_attributes is None:
            print(
                "Need node_file, node_id, and color_attribute to specify continous value"
            )
            exit(1)
        node_colors = get_sequential_node_colors(
            color_list, node_attributes, use_log_scale
        )
    else:
        partition = get_partition(partitions, node_attributes)
        colormap = create_colormap(color_list, partition)
        node_colors = get_node_colors_from_partition(partition, colormap)

    nx_graph = networkx.DiGraph() if arrows else networkx.Graph()
    for n in node_positions:
        nx_graph.add_node(n.node_id)
    if edge_list_file is not None:
        all_edges = read_graph(edge_list_file)
        _add_edges_to_graph(nx_graph, all_edges.edges())

    logger.info(f"writing file: {image_file}")
    layouts.save_graph(
        image_file,
        nx_graph,
        node_positions,
        node_colors=node_colors,
        dpi=dpi,
        light_background=light_background,
        edge_alpha=edge_alpha,
        vertex_alpha=vertex_alpha,
        vertex_line_width=vertex_line_width,
        edge_line_width=edge_line_width,
        figure_width=figure_width,
        figure_height=figure_height,
        vertex_shape=vertex_shape,
        arrows=arrows,
    )


def _add_edges_to_graph(graph, edge_list):
    print(f"graph: edges before {len(graph.edges())}, new edge_list {len(edge_list)}")
    for s, t in edge_list:
        if s in graph and t in graph:
            # only add if both are in the graph, if we only positioned an LCC the whole list could add extra nodes
            graph.add_edge(s, t)
    return graph
