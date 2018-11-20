#!/usr/bin/env python

# utils.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np
import networkx as nx
from functools import reduce

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
    if type(graph) in [nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph]:
        if not is_fully_connected(graph): 
            raise ValueError('Input graph is not fully connected, please use '
                            + 'graspy.utils.find_lcc() to generate a connected graph '
                            + 'before import')
        graph = nx.to_numpy_array(
            graph, nodelist=sorted(graph.nodes), dtype=np.float)
    elif (type(graph) is np.ndarray):
        if len(graph.shape) != 2:
            raise ValueError('Matrix has improper number of dimensions')
        elif graph.shape[0] != graph.shape[1]:
            raise ValueError('Matrix is not square')
        if not np.issubdtype(graph.dtype, np.floating):
            graph = graph.astype(np.float)
        if not is_fully_connected(graph): 
            raise ValueError('Input graph is not fully connected, please use '
                            + 'graspy.utils.find_lcc() to generate a connected graph '
                            + 'before import')
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
    '''
    Checks whether the input graph is fully connected in the undirected case
    or weakly connected in the directed case. 

    Connected means one can get from any vertex u to vertex v by traversing
    the graph. For a directed graph, weakly connected means that the graph 
    is connected after it is converted to an unweighted graph (ignore the 
    direction of each edge)

    Parameters
    ----------
        graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            Input graph in any of the above specified formats. If np.ndarray, 
            interpreted as an n x n adjacency matrix

    Returns
    -------
        boolean: True if the entire input graph is connected

    References
    ----------
        http://mathworld.wolfram.com/ConnectedGraph.html
        http://mathworld.wolfram.com/WeaklyConnectedDigraph.html

    '''
    if type(graph) is np.ndarray:
        if is_symmetric(graph):
            g_object = nx.Graph()
        else:
            g_object = nx.DiGraph()
        graph = nx.from_numpy_matrix(graph, create_using=g_object)
    if type(graph) in [nx.Graph, nx.MultiGraph]:
        return nx.is_connected(graph)
    elif type(graph) in [nx.DiGraph, nx.MultiDiGraph]:
        return nx.is_weakly_connected(graph)

def get_lcc(graph, return_inds=False):
    '''
    Finds the largest connected component for the input graph. 

    The largest connected component is the fully connected subgraph
    which has the most nodes. 

    Parameters
    ----------
        graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            Input graph in any of the above specified formats. If np.ndarray, 
            interpreted as an n x n adjacency matrix
     
        return_inds: boolean, default: False
            Whether to return a np.ndarray containing the indices in the original
            adjacency matrix that were kept and are now in the returned graph.
            Ignored when input is networkx object
    
    Returns
    -------
        graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            New graph of the largest connected component of the input parameter. 

        inds: (optional)
            Indices from the original adjacency matrix that were kept after taking
            the largest connected component 
    '''
    
    input_ndarray = False
    if type(graph) is np.ndarray:
        input_ndarray = True
        if is_symmetric(graph):
            g_object = nx.Graph()
        else:
            g_object = nx.DiGraph()
        graph = nx.from_numpy_matrix(graph, create_using=g_object)
    if type(graph) in [nx.Graph, nx.MultiGraph]:
        graph = max(nx.connected_component_subgraphs(graph), key=len)
    elif type(graph) in [nx.DiGraph, nx.MultiDiGraph]:
        graph = max(nx.weakly_connected_component_subgraphs(graph), key=len)        
    if return_inds:
        nodelist = list(graph.nodes)
    if input_ndarray:
        graph = nx.to_numpy_array(graph)
    if return_inds:
        return graph, nodelist
    return graph

def get_multigraph_lcc(graphs, return_inds=False):
    '''
    Finds the intersection of multiple graphs's largest connected components. 

    Computes the largest connected component for each graph that was input, and 
    takes the intersection over all of these resulting graphs. Note that this 
    does not guarantee finding the largest graph where every node is shared among
    all of the input graphs.

    Parameters
    ----------
        graphs: list or np.ndarray
            if list, each element must be an n x n np.ndarray adjacency matrix
            
     
        return_inds: boolean, default: False
            Whether to return a np.ndarray containing the indices in the original
            adjacency matrix that were kept and are now in the returned graph.
            Ignored when input is networkx object
    
    Returns
    -------
        graph: nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            New graph of the largest connected component of the input parameter. 

        inds: (optional)
            Indices from the original adjacency matrix that were kept after taking
            the largest connected component 
    '''
    lcc_by_graph = []
    inds_by_graph = []
    for graph in graphs:
        lcc, inds = get_lcc(graph, return_inds=True)
        lcc_by_graph.append(lcc)
        inds_by_graph.append(inds)
    inds_intersection = reduce(np.intersect1d, inds_by_graph)
    new_graphs = []        
    for graph in graphs:
        if type(graph) is np.ndarray:
            graph = graph[inds_intersection,:][:,inds_intersection]
        else: 
            graph = graph.subgraph(inds_intersection)
        new_graphs.append(graph)
    if type(graphs) == list:
        return new_graphs
    else:
        return np.stack(new_graphs)