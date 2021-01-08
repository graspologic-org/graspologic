# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import csv
import logging
import networkx
from typing import Any, Dict, List, Tuple

from networkx.generators import community

from graspologic.layouts.nooverlap import remove_overlaps
from graspologic.layouts import NodePosition, render
from graspologic.layouts.nooverlap._node import _Node
from graspologic.layouts.nooverlap._quad_tree import _QuadTree

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s, %(message)s", level=logging.DEBUG
)
logger = logging.getLogger("test.graspologic.layouts.nooverlap")

def _read_input_locs_file(filename, has_header=True):
    with open(filename, "r") as ifile:
        reader = csv.reader(ifile)
        if has_header:
            header = next(reader)
        nodes = []
        colors = {}
        for row in reader:
            nid, x, y, size, community, color = row[:6]
            x = float(x)
            y = float(y)
            size = float(size)
            community = int(community)
            colors[nid] = color
            nodes.append( NodePosition(nid, x, y, size, community))
    return nodes, colors

def graph_from_nodes_only(node_positions: List[NodePosition]):
    graph = networkx.Graph()
    for node in node_positions:
        graph.add_node(node.node_id)
    return graph

def _location(path: str, positions: List[NodePosition], colors: Dict[Any, str]):
    with open(path, "w") as node_positions_out:
        print("id,x,y,size,community,color", file=node_positions_out)
        for position in positions:
            print(
                f"{position.node_id},{position.x},{position.y},{position.size},"
                f"{position.community},{colors[position.node_id]}",
                file=node_positions_out,
            )

def dump_stats_to_csv(stats, filename, header=None):
    with open(filename, "w", encoding="utf-8", newline="") as ofile:
        writer = csv.writer(ofile)
        if header is not None:
            writer.writerow(["node_id"] + header)
        for idx, row in enumerate(stats):
            writer.writerow([idx] + row)

def dump_quad_stats(node_positions : List[NodePosition], csv_filename : str, max_level: int, png_filename: str = None):
    local_nodes = [
        _Node(node.node_id, node.x, node.y, node.size, node.community)
        for node in node_positions
    ]
    qt = _QuadTree(local_nodes, 50)
    stats = qt.get_node_stats(max_level)
    header = qt.get_node_stats_header()
    dump_stats_to_csv(stats, filename=csv_filename, header=header)
    if png_filename is not None:
        draw_quad_tree(stats, png_filename)
    return


def make_graph_and_positions(stats, level=None, vertex_size=20):
    vsize = 20
    graph = networkx.Graph()
    positions = []
    for node_id, row in enumerate(stats):
        min_x, min_y, max_x, max_y = row[3:7]
        print (min_x, min_y, max_x, max_y)
        depth = row[2]
        if level is None:
            graph.add_node(node_id)
            positions.append(NodePosition(node_id=node_id, x=row[0], y=row[1], size=vertex_size, community=depth))
            sw = str(node_id)+'-sw'
            nw = str(node_id)+'-nw'
            ne = str(node_id)+'-ne'
            se = str(node_id)+'-se'
            positions.append(NodePosition(node_id=sw, x=min_x, y=min_y, size=vertex_size, community=depth))
            positions.append(NodePosition(node_id=nw, x=min_x, y=max_y, size=vertex_size, community=depth))
            positions.append(NodePosition(node_id=ne, x=max_x, y=max_y, size=vertex_size, community=depth))
            positions.append(NodePosition(node_id=se, x=max_x, y=min_y, size=vertex_size, community=depth))
            graph.add_edge(sw, nw)
            graph.add_edge(sw, se)
            graph.add_edge(nw, ne)
            graph.add_edge(se, ne)
        elif level == depth:
            graph.add_node(node_id)
            positions.append(NodePosition(node_id=node_id, x=row[0], y=row[1], size=vertex_size, community=depth))
            sw = str(node_id)+'-sw'
            nw = str(node_id)+'-nw'
            ne = str(node_id)+'-ne'
            se = str(node_id)+'-se'
            positions.append(NodePosition(node_id=sw, x=min_x, y=min_y, size=vertex_size, community=depth))
            positions.append(NodePosition(node_id=nw, x=min_x, y=max_y, size=vertex_size, community=depth))
            positions.append(NodePosition(node_id=ne, x=max_x, y=max_y, size=vertex_size, community=depth))
            positions.append(NodePosition(node_id=se, x=max_x, y=min_y, size=vertex_size, community=depth))
            graph.add_edge(sw, nw)
            graph.add_edge(sw, se)
            graph.add_edge(nw, ne)
            graph.add_edge(se, ne)

    return graph, positions

def generate_node_colors(stats: List):
    node_color_map = {}
    color = 'black'
    for node_id, row in enumerate(stats):
        depth = int(row[2])
        node_color_map[node_id] = "red"
        sw = str(node_id)+'-sw'
        nw = str(node_id)+'-nw'
        ne = str(node_id)+'-ne'
        se = str(node_id)+'-se'
        node_color_map[sw] = color
        node_color_map[nw] = color
        node_color_map[ne] = color
        node_color_map[se] = color
    return node_color_map


def draw_quad_tree(stats: List, filename: str, dpi=500):
    graph, positions = make_graph_and_positions(stats)
    node_colors = generate_node_colors(stats)
    print (f"num node colors: {len(node_colors)}")
    render.save_graph(filename, graph, positions, node_colors=node_colors, dpi=dpi, edge_alpha=0.8)

    return



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_locs", help="CSV file in the format of (id,x,y,size,community,color) with the first line a header", required=True)
    parser.add_argument("--image_file", help='Name for final Image file', required=False, default=None)
    parser.add_argument("--location_file", help='Name for final location file', required=False, default=None)
    parser.add_argument("--quad_dump", help='File name for CSV dump of quad tree', required=False, default=None)
    parser.add_argument("--quad_png", help='File name for PNG dump of quad tree', required=False, default=None)
    parser.add_argument("--dpi", help='Only used if --image_file is specified', type=int, required=False, default=500 )
    args = parser.parse_args()


    node_positions, node_colors = _read_input_locs_file(args.input_locs)
    logger.debug(f"read nodes: {len(node_positions)}, read colors: {len(node_colors)}")

    new_positions = remove_overlaps(node_positions)
    graph = graph_from_nodes_only(new_positions)

    if args.image_file is not None:
        render.save_graph(args.image_file, graph, new_positions, node_colors=node_colors, dpi=args.dpi)
    if args.location_file is not None:
        _location(args.location_file, new_positions, node_colors)
    if args.quad_dump is not None:
        dump_quad_stats(node_positions, args.quad_dump, png_filename=args.quad_png, max_level=50)
    #if args.quad_png is not None:
    #    draw_quad_tree()



if __name__ == "__main__":
    main()
