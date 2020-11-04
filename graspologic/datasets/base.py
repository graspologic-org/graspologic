# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from os.path import dirname, join
import numpy as np


def load_drosophila_left(return_labels=False):
    """
    Load the left Drosophila larva mushroom body connectome

    The mushroom body is a learning and memory center in the fly
    brain which is involved in sensory integration and processing.
    This connectome was observed by electron microscopy and then
    individial neurons were reconstructed; synaptic partnerships
    between these neurons became the edges of the graph.

    Parameters
    ----------
    return_labels : bool, optional (default=False)
        whether to have a second return value which is an array of
        cell type labels for each node in the adjacency matrix

    Returns
    -------
    graph : np.ndarray
        Adjacency matrix of the connectome
    labels : np.ndarray
        Only returned if ``return_labels`` is true. Array of
        string labels for each cell (vertex)

    References
    ----------
    .. [1] Eichler, K., Li, F., Litwin-Kumar, A., Park, Y., Andrade, I.,
           Schneider-Mizell, C. M., ... & Fetter, R. D. (2017). The
           complete connectome of a learning and memory centre in an insect
           brain. Nature, 548(7666), 175.
    """

    module_path = dirname(__file__)
    folder = "drosophila"
    filename = "left_adjacency.csv"
    with open(join(module_path, folder, filename)) as csv_file:
        graph = np.loadtxt(csv_file, dtype=int)
    if return_labels:
        filename = "left_cell_labels.csv"
        with open(join(module_path, folder, filename)) as csv_file:
            labels = np.loadtxt(csv_file, dtype=str)
        return graph, labels
    else:
        return graph


def load_drosophila_right(return_labels=False):
    """
    Load the right Drosophila larva mushroom body connectome

    The mushroom body is a learning and memory center in the fly
    brain which is involved in sensory integration and processing.
    This connectome was observed by electron microscopy and then
    individial neurons were reconstructed; synaptic partnerships
    between these neurons became the edges of the graph.

    Parameters
    ----------
    return_labels : bool, optional (default=False)
        whether to have a second return value which is an array of
        cell type labels for each node in the adjacency matrix

    Returns
    -------
    graph : np.ndarray
        Adjacency matrix of the connectome
    labels : np.ndarray
        Only returned if `return_labels` is true. Array of
        string labels for each cell (vertex)

    References
    ----------
    .. [1] Eichler, K., Li, F., Litwin-Kumar, A., Park, Y., Andrade, I.,
           Schneider-Mizell, C. M., ... & Fetter, R. D. (2017). The
           complete connectome of a learning and memory centre in an insect
           brain. Nature, 548(7666), 175.
    """

    module_path = dirname(__file__)
    folder = "drosophila"
    filename = "right_adjacency.csv"
    with open(join(module_path, folder, filename)) as csv_file:
        graph = np.loadtxt(csv_file, dtype=int)
    if return_labels:
        filename = "right_cell_labels.csv"
        with open(join(module_path, folder, filename)) as csv_file:
            labels = np.loadtxt(csv_file, dtype=str)
        return graph, labels
    else:
        return graph
