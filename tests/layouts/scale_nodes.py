# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import csv
import logging
import networkx
import numpy
from typing import List, AnyStr

from graspologic.layouts import NodePosition, render
from graspologic.layouts.auto import _scale_points, _covered_size
from graspologic.layouts.nooverlap._quad_tree import _QuadTree

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s, %(message)s", level=logging.DEBUG
)
logger = logging.getLogger("test.graspologic.layouts.nooverlap")

def _graph_from_positions(
    node_positions: List[NodePosition],
) -> networkx.Graph:
    graph = networkx.Graph()
    for np in node_positions:
        graph.add_node(np.node_id, x=np.x, y=np.y, size=np.size, community=np.community)
    return graph

def _positions_and_sizes(filename: AnyStr, skip_header: bool):
    sizes = {}
    node_colors = {}
    positions = []
    with open(filename, "r") as location_io:
        if skip_header:
            next(location_io)
        for line in location_io:
            node_id, x, y, size, community, color = line.strip().split(",")
            positions.append(
                NodePosition(node_id, float(x), float(y), float(size), int(community))
            )
            sizes[node_id] = float(size)
            node_colors[node_id] = color
    return positions, sizes, node_colors

def _extract_positions_as_numpy_array(node_pos_list: List[NodePosition]):
    positions = []
    print (f"size of np: {len(node_pos_list)}")
    for n in node_pos_list:
        positions.append([n.x, n.y])
    retval = numpy.array(positions)
    print (f"{retval.shape}")
    return retval
def _np_positions_to_node_positions(scaled : numpy.array, old_positions: List[NodePosition]):
    new_positions = []
    for idx, np in enumerate(old_positions):
        new_positions.append( NodePosition(np.node_id, scaled[idx][0], scaled[idx][1], np.size, np.community))
    return new_positions

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="input file with node sizes and locations", required=True)
    parser.add_argument("--output", help='embedding type', required=True)
    parser.add_argument("--density", type=float, help='how much to Scale the nodes', required=False, default=1.0)
    args = parser.parse_args()

    input_file = args.input
    out_file = args.output
    density = args.density
    dpi=800
    positions, sizes, colors = _positions_and_sizes(input_file, True)
    graph = _graph_from_positions(positions)
    np_positions = _extract_positions_as_numpy_array(positions)
    print (f"shape: {np_positions.shape}")
    covered_size = _covered_size(sizes)
    print (f"canvas: {covered_size}, density: {density}")
    scaled_positions = _scale_points(np_positions, covered_size, density)
    new_node_positions = _np_positions_to_node_positions(scaled_positions, positions)
    render.save_graph(out_file, graph, new_node_positions, node_colors=colors)


if __name__ == "__main__":
    main()
