# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
import sys
import networkx as nx

from graspologic.layouts._helpers import ensure_directory_for_file

# from graspologic.layouts import (
#     layout_node2vec_umap_from_file,
#     layout_node2vec_tsne_from_file,
# )

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s, %(message)s", level=logging.INFO
)
logger = logging.getLogger("graspologic_layouts")


def _graph_from_file(
    path: str,
    skip_header: bool = False,
) -> nx.Graph:
    # loads an edge list from a file into a networkx graph
    pass


def valid_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="python -m graspologic.layouts",
        description="Runnable module that automatically generates a layout of a graph "
        "by a provided edge list",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--edge_list",
        help="edge list in csv file. must be source,target,weight.",
        required=True,
    )
    parser.add_argument(
        "--skip_header",
        help="skip first line in csv file, corresponding to header.",
        required=False,
        default=False,
    )
    parser.add_argument(
        "--layout_type",
        help="2-dimensional umap or tsne downprojection of node2vec embedding space for layout purposes",
        required=False,
        default="n2vumap",
        choices=["n2vumap", "n2vtsne"],
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
    args = parser.parse_args()

    if args.image_file is None and args.location_file is None:
        print(
            "error: --image_file or --location_file must be provided", file=sys.stderr
        )
        exit(-1)

    return args


arguments = valid_args()

if arguments.layout_type == "n2vumap":
    layout_node2vec_umap_from_file(
        arguments.edge_list_file,
        arguments.image_file,
        arguments.location_file,
        arguments.dpi,
        arguments.skip_headers,
        arguments.max_edges,
    )
elif arguments.layout_type == "n2vtsne":
    layout_node2vec_tsne_from_file(
        arguments.edge_list_file,
        arguments.image_file,
        arguments.location_file,
        arguments.dpi,
        arguments.skip_headers,
        arguments.max_edges,
    )
