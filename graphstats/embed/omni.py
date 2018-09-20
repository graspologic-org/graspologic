# omni.py
# Created by Jaewon Chung on 2018-09-10.
# Email: j1c@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np
from embed import BaseEmbedder
from utils import import_graph


def _check_valid_graphs(graphs):
    """
    Raises an ValueError if all items in the graphs have same
    number of vertices and are square.

    Parameters
    ----------
    graphs : list of graphs
        List of array-like, (n_vertices, n_vertices) or list of 
        networkx.Graph.

    Raises
    ------
    ValueError
        If all graphs do not have same number of vertices and are square.
    """
    shapes = set(map(np.shape, graphs))

    if len(shapes) > 1:
        msg = "There are {} different sizes of graphs.".format(len(shapes))
        raise ValueError(msg)
    elif len(shapes) == 0:
        msg = "No input graphs found."
        raise ValueError(msg)


def _get_omni_matrix(graphs):
    """
    Helper function for creating the mnibus matrix.

    Parameters
    ----------
    graphs : list of graphs
        List of array-like, (n_vertices, n_vertices), or list of 
        networkx.Graph.

    Returns
    -------
    out : 2d-array
        Array of shape (n_vertices * n_graphs, n_vertices * n_graphs)
    """
    n = len(graphs)

    out = (np.tile(np.hstack(graphs),
                   (n, 1)) + np.tile(np.vstack(graphs), (1, n))) / 2

    return out


class OmnibusEmbed(BaseEmbed):
    """
    Omnibus embedding of arbitrary number of input graphs with matched vertex 
    sets.

    Parameters
    ----------
    method: object (default selectSVD)
        the method to use for dimensionality reduction.
    args: list, optional (default None)
        options taken by the desired embedding method as arguments.
    kwargs: dict, optional (default None)
        options taken by the desired embedding method as key-worded
        arguments.

    See Also
    --------
    graphstats.embed.svd.SelectSVD, graphstats.embed.svd.selectDim
    """

    def __init__(self, method=selectSVD, *args, **kwargs):
        super().__init__(method=selectSVD, *args, **kwargs)

    def fit(self, graphs):
        """
		Omnibus embedding 

		Parameters
		----------
        graphs : list of graphs
            List of array-like, (n_vertices, n_vertices), or list of 
            networkx.Graph.

		Returns
		-------
        X : array-like, shape (n_vertices, k)
            The estimated latent positions.
        Y : array-like, shape (n_vertices, k)
            If graph is not symmetric, the  right estimated latent
            positions. if graph is symmetric, "None".
		"""
        # TODO: Convert networkx.Graph to np.arrays if Graphs are given.
        _check_valid_graphs(graphs)

        omni_matrix = _get_omni_matrix(graphs)

        return omni_matrix
