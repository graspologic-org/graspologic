
import csv
import argparse
from graspologic.layouts.nooverlap import _node

parser = argparse.ArgumentParser()
parser.add_argument("--infile", help='CSV input file', required=True)
#parser.add_argument("--outfile", help='CSV file name', required=True)
args = parser.parse_args()

infile = args.infile
#outfile = args.outfile


def read_graph(infile):
	with open(infile, 'r', encoding='utf-8') as ifile:
		reader = csv.reader(ifile)
		hash_graph = {}
		for nid,xcoord,ycoord,size,communityid,color in reader:
			#hash_graph[nid] = [float(xcoord), float(ycoord), float(size), communityid, color]
			hash_graph[nid] = _node(nid, xcoord, ycoord, size, communityid, color)
	return hash_graph



graph = read_graph(infile)

sg = sorted(graph.values(), key=lambda x: x.size)

for idx, n in enumerate(sg):
	print (idx, n.size)
