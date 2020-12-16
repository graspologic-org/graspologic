# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
from pathlib import Path
import sys
from typing import List

import networkx as nx

from . import auto, NodePosition
from . import render


def _graph_from_file(
    path: str,
    skip_header: bool = False,
) -> nx.Graph:
    graph = nx.Graph()
    with open(path, "r") as edge_io:
        if skip_header is True:
            next(edge_io)
        for line in edge_io:
            source, target, weight = line.strip().split(",")
            weight = float(weight)
            if graph.has_edge(source, target):
                weight += graph[source][target]["weight"]
            graph.add_edge(source, target, weight=weight)
    return graph


def _common_edge_list_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--edge_list",
        help="edge list in csv file. must be source,target,weight.",
        required=True,
    )
    parser.add_argument(
        "--skip_header",
        help="skip first line in csv file, corresponding to header.",
        type=bool,
        required=False,
        default=False,
    )
    parser.add_argument(
        "--image_file",
        help="output path and filename for generated image file. "
             "required if --location_file is omitted.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--location_file",
        help="output path and filename for location file. "
             "required if --image_file is omitted.",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--max_edges",
        help="maximum edges to keep during embedding. edges with low weights will be "
             "pruned to keep at most this many edges",
        type=int,
        required=False,
        default=10000000,
    )
    parser.add_argument(
        "--dpi",
        help="used with --image_file to render an image at this dpi",
        type=int,
        required=False,
        default=500,
    )
    return parser


def _ensure_output_dir(path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _render(path: str, graph: nx.Graph, positions: List[NodePosition]):
    # todo: colormap properly
    colors = {position.node_id: "#000000" for position in positions}
    render.save_graph(path, graph, positions, node_colors=colors)


def _location(path: str, positions: List[NodePosition]):
    with open(path, "w") as node_positions_out:
        print("id,x,y,size,community", file=node_positions_out)
        for position in positions:
            print(
                f"{position.node_id},{position.x},{position.y},{position.size},"
                f"{position.community}", file=node_positions_out
            )


def _output(
    arguments: argparse.Namespace,
    graph: nx.Graph,
    positions: List[NodePosition]
):
    if arguments.image_file is not None:
        _render(arguments.image_file, graph, positions)
    if arguments.location_file is not None:
        _location(arguments.location_file, positions)


def _tsne(arguments: argparse.Namespace):
    valid_args(arguments)
    graph = _graph_from_file(arguments.edge_list, arguments.skip_header)
    graph, positions = auto.layout_tsne(
        graph,
        perplexity=30,
        n_iter=1000,
        max_edges=arguments.max_edges
    )
    _output(arguments, graph, positions)


def _umap(arguments: argparse.Namespace):
    valid_args(arguments)
    graph = _graph_from_file(arguments.edge_list, arguments.skip_header)
    graph, positions = auto.layout_umap(graph, max_edges=arguments.max_edges)
    _output(arguments, graph, positions)


def _render(arguments: argparse.Namespace):
    pass


def _parser() -> argparse.ArgumentParser:
    root_parser = argparse.ArgumentParser(
        prog="python -m graspologic.layouts",
        description="Runnable module that automatically generates a layout of a graph "
                    "by a provided edge list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    root_parser.add_argument(
        "--verbose",
        type=bool,
        required=False,
        default=False,
    )

    subparsers = root_parser.add_subparsers(required=True)

    n2vumap_parser = subparsers.add_parser(
        "n2vumap",
        help="Auto layout using UMAP for dimensionality reduction",
    )
    _common_edge_list_args(n2vumap_parser)
    n2vumap_parser.set_defaults(func=_umap)
    n2vtsne_parser = subparsers.add_parser(
        "n2vtsne",
        help="Auto layout using tSNE for dimensionality reduction",
    )
    _common_edge_list_args(n2vtsne_parser)
    n2vtsne_parser.set_defaults(func=_tsne)
    render_parser = subparsers.add_parser(
        "render",
        help="Renders a graph via an input file",
    )
    render_parser.set_defaults(func=_render)
    return root_parser


def valid_args(args: argparse.Namespace):
    if args.image_file is None and args.location_file is None:
        print(
            "error: --image_file or --location_file must be provided", file=sys.stderr
        )
        exit(-1)
    if args.location_file is not None:
        _ensure_output_dir(args.location_file)
    if args.image_file is not None:
        _ensure_output_dir(args.image_file)


parser = _parser()

args = parser.parse_args()
if args.verbose is True:
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s:%(name)s, %(message)s", level=logging.INFO
    )
    logger = logging.getLogger("graspologic.layouts")

args.func(args)
