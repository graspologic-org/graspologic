# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging

logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s, %(message)s", level=logging.INFO
)
logger = logging.getLogger("graspologic_layouts")


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_list",
        help="CSV file in the formate of (source,target,weight) with the first line a header",
        required=True,
    )
    parser.add_argument(
        "--layout_type",
        help='Valid types "autolayout", "n2vumap" (default), and "n2vtsne", others may be added',
        required=False,
        default="n2vumap",
    )
    parser.add_argument(
        "--image_file", help="Name for final Image file", required=False, default=None
    )
    parser.add_argument(
        "--location_file",
        help="Name for final location file",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--max_edges",
        help="Maximum edges to keep during embedding (default 10000000)",
        type=int,
        required=False,
        default=10000000,
    )
    parser.add_argument(
        "--dpi",
        help="Only used if --image_file is specified",
        type=int,
        required=False,
        default=500,
    )
    args = parser.parse_args()

    edge_list_file = args.edge_list
    layout_type = args.layout_type
    image_file = args.image_file
    location_file = args.location_file
    max_edges = args.max_edges
    dpi = args.dpi

    # moved import after command line parsing to speed up command line failures
    from ._helpers import read_node_file, ensure_directory_for_file
    from .layout_from_edges import (
        layout_node2vec_umap_from_file,
        layout_node2vec_tsne_from_file,
    )

    if image_file is None and location_file is None:
        print(f"Must specify an image file, a location file or both")
        return
    ensure_directory_for_file(image_file)
    ensure_directory_for_file(location_file)

    if layout_type == "n2vumap":
        layout_node2vec_umap_from_file(
            edge_list_file, image_file, location_file, dpi, max_edges
        )
    elif layout_type == "n2vtsne":
        layout_node2vec_tsne_from_file(
            edge_list_file, image_file, location_file, dpi, max_edges
        )
    # 	elif layout_type == 'autolayout':
    # 		autolayout(edge_list_file, image_file, location_file, dpi)
    else:
        print(f"Invalid Layout type specified {layout_type}")
    return


_main()
