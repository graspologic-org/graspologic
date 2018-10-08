#!/usr/bin/env python

# utils.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np
import networkx as nx
from scipy.stats import rankdata 

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
    graph = import_graph(graph)
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

    Currently only supports normalized laplacian.

    Parameters
    ----------
        graph: object
            Either array-like, (n_vertices, n_vertices) numpy array,
            or an object of type networkx.Graph.

    Returns
    -------
        L: numpy.ndarray
            2D (n_vertices, n_vertices) array representing graph 
            laplacian of specified form
    """
    adj_matrix = import_graph(graph)

    if form == 'I-DAD':
        D_vec = np.sum(adj_matrix, axis=0)
        D_root = np.diag(D_vec ** -0.5)
        L = np.diag(D_vec) - adj_matrix
        L = np.dot(D_root, L)
        L = np.dot(L, D_root)
        # L = D_root @ L @ D_root # not compatible with python 3.4
    else: 
        raise TypeError('Unsuported Laplacian normalization')

    return L

def pass_to_ranks(graph, method='zero-boost'):
    """
    Rescales edge weights of an adjacency matrix based on their relative rank in 
    the graph. 

    Parameters
    ----------
        graph: Adjacency matrix 
        
        method: string, optional
            'zero-boost' preserves the edge weight for all 0s, but ranks the other
            edges as if the ranks of all 0 edges has been assigned. If there are 
            10 0-valued edges, the lowest non-zero edge gets weight 11 / (number
            of possible edges). Ties settled by the average of the weight that those
            edges would have received. Number of possible edges is determined 
            by the type of graph (loopless or looped, directed or undirected)

    """ 
    
    graph = import_graph(graph)

    # do nothing if the graph is unweighted
    if is_unweighted(graph):
        return graph
    else: 
        if method == 'zero-boost':
            if is_symmetric(graph):
                # start by working with half of the graph, since symmetric
                triu = np.triu(graph)
                non_zeros = triu[triu != 0]
                rank = rankdata(non_zeros)
                
                num_zeros = 0
                possible_edges = 0
                if is_loopless(graph):
                    num_zeros = (len(graph[graph == 0]) - graph.shape[0])/2
                    possible_edges = graph.shape[0] * (graph.shape[0] - 1) / 2 
                else: 
                    num_zeros = (len(triu[triu == 0]) - graph.shape[0] * (graph.shape[0] - 1) / 2) 
                    possible_edges = graph.shape[0] * (graph.shape[0] + 1) / 2
                
                # shift up by the number of zeros 
                rank = rank + num_zeros

                # normalize by the number of possible edges for this kind of graph
                rank = rank / possible_edges

                # put back into matrix form and reflect over the diagonal
                triu[triu != 0] = rank 
                graph = symmetrize(triu, method='triu')
                
                return graph
            else: 
                raise NotImplementedError()

        else: 
            raise ValueError('Unsuported pass-to-ranks method')

#     if is_symmetric(graph): 

#         print(graph.astype(int))
#         rank = rankdata(graph)
#         print(np.reshape(rank, (graph.shape[0], graph.shape[1])))
#         rank = np.reshape(rank, (graph.shape[0], graph.shape[1])) - rank.min()
#         print(rank)
#         rank = 2 * rank / (graph_nonzero.shape[0] + 1)
#         print(rank)