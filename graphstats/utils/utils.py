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

def check_square(adj_mat):
	"""
	Function to ensure adjacency matrix is square

	Parameters:
	----------
		adj_mat: numpy matrix

	Raises:
	------
	ValueError
		If the matrix is not square or has more than 2 dimensions
	"""
	# Believe this is unnecessary based on import func 
	# if type(adj_mat) != np.ndarray:
	# 	raise TypeError

	shape = adj_mat.shape
	if len(adj_mat.shape) != 2:
		raise ValueError('Matrix has improper number of dimensions')
	elif adj_mat[0] != adj_mat[1]:
		raise ValueError('Matrix is not square')
	