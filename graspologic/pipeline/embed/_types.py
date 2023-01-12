from typing import Union

import networkx as nx
from pkg_resources import parse_version

major = parse_version(nx.__version__).major

if major >= 3:
    NxGraphType = Union[nx.Graph, nx.DiGraph]
else:
    NxGraphType = Union[nx.Graph, nx.DiGraph, nx.OrderedGraph, nx.OrderedDiGraph]
