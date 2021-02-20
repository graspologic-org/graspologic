# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from os.path import dirname, join
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.utils import Bunch

from ..utils import import_edgelist


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


def load_mice():
    """
    Load connectomes of mice from distinct genotypes.

    Dataset of 32 mouse connectomes derived from whole-brain diffusion
    magnetic resonance imaging of four distinct mouse genotypes:
    BTBR T+ Itpr3tf/J (BTBR), C57BL/6J(B6), CAST/EiJ (CAST), and DBA/2J (DBA2).
    For each strain, connectomes were generated from eight age-matched mice
    (N = 8 per strain), with a sex distribution of four males and four females.
    Each connectome was parcellated using asymmetric Waxholm Space, yielding a
    vertex set with a total of 332 regions of interest (ROIs) symmetrically
    distributed across the left and right hemispheres. Within a given
    hemisphere, there are seven superstructures consisting up multiple ROIs,
    resulting in a total of 14 distinct communities in each connectome.

    Returns
    -------
    data : :class:`~sklearn.utils.Bunch`
        Dictionary-like object, with the following attributes.

        graphs : list of np.ndarray
            List of adjacency matrices of the connectome
        labels : np.ndarray
            Array of string labels for each mouse (subject)
        atlas : pd.DataFrame
            DataFrame of information for each ROI
        blocks : pd.DataFrame
            DataFrame of block assignments for each ROI
        features : pd.DataFrame
            DataFrame of anatomical features for each ROI in each connectome
        participants : pd.DataFrame
            DataFrame of subject IDs and genotypes for each connectome
        meta : Dictionary
            Dictionary with meta information about the dataset (n_subjects and n_vertices)

    References
    ----------
    .. [1] Wang, N., Anderson, R. J., Ashbrook, D. G., Gopalakrishnan, V.,
           Park, Y., Priebe, C. E., ... & Johnson, G. A. (2020). Variability
           and heritability of mouse brain structure: Microscopic MRI atlases
           and connectomes for diverse strains. NeuroImage.
           https://doi.org/10.1016/j.neuroimage.2020.117274
    """

    data = Path(__file__).parent.joinpath("mice")

    # Load all connectomes and construct a dictionary of study metadata
    graphs, vertices = import_edgelist(data.joinpath("edgelists"), return_vertices=True)

    n_vertices = len(vertices)
    n_subjects = len(graphs)
    meta = {"n_subjects": n_subjects, "n_vertices": n_vertices}

    # Read the participants file and get genotype labels
    participants = pd.read_csv(data.joinpath("participants.csv"))
    labels = participants["genotype"].values

    # Read the atlas and block information
    atlas = pd.read_csv(data.joinpath("atlas.csv"))
    blocks = pd.read_csv(data.joinpath("blocks.csv"))

    # Read features
    tmp = []
    for fl in data.joinpath("features").glob("*" + "csv"):
        subid = fl.stem
        df = pd.read_csv(fl, skiprows=2)
        df["participant_id"] = subid
        tmp.append(df)
    features = pd.concat(tmp, axis=0)
    features = features.reset_index(drop=True)

    return Bunch(
        graphs=graphs,
        labels=labels,
        atlas=atlas,
        blocks=blocks,
        features=features,
        participants=participants,
        meta=meta,
    )
