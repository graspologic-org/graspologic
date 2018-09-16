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
		graph: array-like, shape (n_vertices, n_vertices)
		 A graph.
		 
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

def is_symmetric(X):
	if np.array_equal(X, X.T):
		return True
	else:
		return False

def symmetrize(graph, method='triu'):
	"""
	A function for forcing symmetry upon a graph.

	Parameters
	----------
		graph: object
			Either array-like, (n_vertices, n_vertices) numpy matrix,
			or an object of type networkx.Graph.
		method: string
			An option indicating which half of the edges to
			retain when symmetrizing. Options are 'triu' for retaining
			the upper right triangle, 'tril' for retaining the lower
			left triangle, or 'avg' to retain the average weight between the
			upper and lower right triangle, of the adjacency matrix.

	Returns
	-------
		graph: array-like, shape(n_vertices, n_vertices)
	"""
	graph = import_graph(graph)
	if method is 'triu':
		graph = np.triu(graph)
	elif method is 'tril':
		graph = np.tril(graph)
	elif method is 'avg':
		graph = (np.triu(graph) + np.tril(graph))/2
	else:
		msg = "You have not passed a valid parameter for the method."
		raise ValueError(msg)
	# A = A + A' - diag(A)
	graph = graph + graph.T - np.diag(graph)
	return(graph)