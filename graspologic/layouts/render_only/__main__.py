# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import argparse
import logging
from graspologic.layouts.render_only.render_graph import render_graph_from_files


logging.basicConfig(
    format="%(asctime)s:%(levelname)s:%(name)s, %(message)s", level=logging.INFO
)
logger = logging.getLogger("graspologic_layouts.render_only")


color_file_json = "colors-100.json"


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--edge_list",
        help="Only required if you want to show edges CSV file in the formate of (source,target,weight) with the first line a header",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--location_file",
        help="Name of final location file",
        required=True,
        default=None,
    )
    parser.add_argument(
        "--node_file",
        help="CSV file in the formate of (nid,attribute1,attribute2) with the first line a header",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--color_file",
        help="JSON color file from thematic",
        required=False,
        default=color_file_json,
    )
    parser.add_argument(
        "--node_id",
        help="Label of Node ID attribute in the node file",
        required=False,
        default="Id",
    )
    parser.add_argument(
        "--color_attribute",
        help="Label of attribute in the node file to use for coloring",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--image_file", help="Name for final Image file", required=True, default=None
    )
    parser.add_argument(
        "--dark_background", dest="light_background", action="store_false", default=True
    )
    parser.add_argument(
        "--color_is_continuous", dest="continuous", action="store_true", default=False
    )
    parser.add_argument(
        "--color_scheme", help="color scheme from color file to use", default=None
    )
    parser.add_argument(
        "--use_log_scale_for_color",
        dest="use_log_scale",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--vertex_alpha",
        help="Alpha (transparency) of vertices in visualization (default 0.9)",
        type=float,
        required=False,
        default=0.9,
    )
    parser.add_argument(
        "--vertex_line_width",
        help="Line width of vertex outline (default 0.01)",
        type=float,
        required=False,
        default=0.01,
    )
    parser.add_argument(
        "--edge_line_width",
        help="Line width of edge (default 0.5)",
        type=float,
        required=False,
        default=0.5,
    )
    parser.add_argument(
        "--edge_alpha",
        help="Alpha (transparency) of edges in visualization (default 0.2)",
        type=float,
        required=False,
        default=0.2,
    )
    parser.add_argument(
        "--figure_width",
        help="Width of figure (default 15.0)",
        type=float,
        required=False,
        default=15.0,
    )
    parser.add_argument(
        "--figure_height",
        help="Height of figure (default 15.0)",
        type=float,
        required=False,
        default=15.0,
    )
    parser.add_argument(
        "--vertex_shape",
        help="Matplotlib Marker for the vertex shape.",
        required=False,
        default="o",
    )
    parser.add_argument("--arrows", dest="arrows", action="store_true", default=False)
    parser.add_argument(
        "--dpi", help="Seth DPI for image", type=int, required=False, default=500
    )
    args = parser.parse_args()

    edge_list_file = args.edge_list
    location_file = args.location_file
    node_file = args.node_file
    image_file = args.image_file
    color_file = args.color_file
    node_id = args.node_id
    color_attribute = args.color_attribute
    light_background = args.light_background
    color_is_continuous = args.continuous
    color_scheme = args.color_scheme
    use_log_scale = args.use_log_scale
    vertex_alpha = args.vertex_alpha
    vertex_line_width = args.vertex_line_width
    edge_line_width = args.edge_line_width
    edge_alpha = args.edge_alpha
    figure_width = args.figure_width
    figure_height = args.figure_height
    vertex_shape = args.vertex_shape
    arrows = args.arrows
    dpi = args.dpi

    if color_scheme is None:
        color_scheme = "sequential" if color_is_continuous else "nominal"
    logger.info(f"color scheme: {color_scheme}")

    render_graph_from_files(
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
    )


_main()
