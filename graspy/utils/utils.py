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

	Parameters
	----------
    graph: object
        Either array-like, shape (n_vertices, n_vertices) numpy array,
        or an object of type networkx.Graph.

	Returns
	-------
    graph: array-like, shape (n_vertices, n_vertices)
        A graph.
		 
	See Also
	--------
		networkx.Graph, numpy.array
	"""
    if type(graph) is nx.Graph:
        graph = nx.to_numpy_array(
            graph, nodelist=sorted(graph.nodes), dtype=np.float)
    elif (type(graph) is np.ndarray):
        if len(graph.shape) != 2:
            raise ValueError('Matrix has improper number of dimensions')
        elif graph.shape[0] != graph.shape[1]:
            raise ValueError('Matrix is not square')
        
        if not np.issubdtype(graph.dtype, np.floating):
            graph = graph.astype(np.float)

    else:
        msg = "Input must be networkx.Graph or np.array, not {}.".format(
            type(graph))
        raise TypeError(msg)
    if not is_fully_connected(graph): 
        raise UserWarning('WARNING: the graph that has been input ' 
                          + 'is not fully connected, GraSPy functions ' 
                          + 'may not work as expected')
    return graph


def is_symmetric(X):
    return np.array_equal(X, X.T)

def is_loopless(X):
    return not np.any(np.diag(X) != 0)
    
def is_unweighted(X): 
    return ((X==0) | (X==1)).all()

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
            the graph with asymmetries removed.
    """
    # graph = import_graph(graph)
    if method is 'triu':
        graph = np.triu(graph)
    elif method is 'tril':
        graph = np.tril(graph)
    elif method is 'avg':
        graph = (np.triu(graph) + np.tril(graph)) / 2
    else:
        msg = "You have not passed a valid parameter for the method."
        raise ValueError(msg)
    # A = A + A' - diag(A)
    graph = graph + graph.T - np.diag(np.diag(graph))
    return graph


def remove_loops(graph):
    """
    A function to remove loops from a graph.

    Parameters
    ----------
        graph: object
            Either array-like, (n_vertices, n_vertices) numpy matrix,
            or an object of type networkx.Graph.

    Returns
    -------
        graph: array-like, shape(n_vertices, n_vertices)
            the graph with self-loops (edges between the same node) removed.
    """
    graph = import_graph(graph)
    graph = graph - np.diag(np.diag(graph))
    
    return graph

def to_laplace(graph, form='I-DAD'):
    """
    A function to convert graph adjacency matrix to graph laplacian. 

    Currently supports I-DAD and DAD laplacians, where D is the diagonal
    matrix of degrees of each node raised to the -1/2 power, I is the 
    identity matrix, and A is the adjacency matrix

    Parameters
    ----------
        graph: object
            Either array-like, (n_vertices, n_vertices) numpy array,
            or an object of type networkx.Graph.

        form: string
            I-DAD: computes L = I - D*A*D
            DAD: computes L = D*A*D

    Returns
    -------
        L: numpy.ndarray
            2D (n_vertices, n_vertices) array representing graph 
            laplacian of specified form
    """
    valid_inputs = ['I-DAD', 'DAD']
    if form not in valid_inputs:
        raise TypeError('Unsuported Laplacian normalization')
    adj_matrix = import_graph(graph)
    if not is_symmetric(adj_matrix):
        raise ValueError('Laplacian not implemented/defined for directed graphs')
    D_vec = np.sum(adj_matrix, axis=0)
    D_root = np.diag(D_vec ** -0.5)
    
    if form == 'I-DAD':
        L = np.diag(D_vec) - adj_matrix
        L = np.dot(D_root, L)
        L = np.dot(L, D_root)
    elif form == 'DAD':
        L = np.dot(D_root, adj_matrix)
        L = np.dot(L, D_root)
    
    return L

def is_fully_connected(graph):
    # remove loops to evaluate in/out degree just by summing
    graph = graph - np.diag(np.diag(graph))
    left_degree = np.sum(graph, axis=0)
    right_degree = np.sum(graph, axis=1)
    left_zeros = np.where(left_degree == 0)
    right_zeros = np.where(right_degree == 0)
    both = np.intersect1d(left_zeros, right_zeros)
    return len(both) == 0