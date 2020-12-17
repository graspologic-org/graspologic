# import csv
# import argparse
# from graspologic.layouts.nooverlap import _node
#
# parser = argparse.ArgumentParser()
# parser.add_argument("--infile", help="CSV input file", required=True)
# parser.add_argument("--outfile", help="CSV file name", required=True)
# parser.add_argument(
#     "--scale", help="Scale Percent", type=float, required=False, default=1.0
# )
# args = parser.parse_args()
#
# infile = args.infile
# outfile = args.outfile
# scale = args.scale
#
#
# def read_graph(infile):
#     with open(infile, "r", encoding="utf-8") as ifile:
#         reader = csv.reader(ifile)
#         hash_graph = {}
#         for nid, xcoord, ycoord, size, communityid, color in reader:
#             # hash_graph[nid] = [float(xcoord), float(ycoord), float(size), communityid, color]
#             hash_graph[nid] = _node(nid, xcoord, ycoord, size, communityid, color)
#     return hash_graph
#
#
# def write_graph(nodes, outfile):
#     with open(outfile, "w", encoding="utf-8", newline="") as ofile:
#         writer = csv.writer(ofile)
#         for n in nodes:
#             writer.writerow(n.to_list())
#     return
#
#
# def scale_node_size(nodes, scale_factor):
#     for nid, n in nodes.items():
#         n.size = n.size * scale_factor
#     return nodes
#
#
# graph = read_graph(infile)
# graph = scale_node_size(graph, scale)
# write_graph(graph.values(), outfile)
