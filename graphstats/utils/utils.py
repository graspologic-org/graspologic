#!/usr/bin/env python

# utils.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np
import networkx as nx


def import_graph(graph):
	"""
	A function for reading a graph and returning a shared
	data type. Makes IO cleaner and easier.

	Parameters:
	-----------
		graph: object
		 Either array-like, (n_vertices, n_vertices) numpy matrix,
		 or an object of type networkx.Graph.

	Returns:
	--------
		graph: object
		 A consistent data model for the package.
		 
	See Also:
	---------
		networkx.Graph, numpy.array
	"""
	if type(graph) is nx.Graph:
		graph = nx.to_numpy_matrix(graph)
	elif (type(graph) is np.ndarray):
		pass
	else:
		raise TypeError
	return graph
