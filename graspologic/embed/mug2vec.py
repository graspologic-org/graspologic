# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from sklearn.base import BaseEstimator

from .mds import ClassicalMDS
from .omni import OmnibusEmbed
from ..utils import pass_to_ranks


class mug2vec(BaseEstimator):
    r"""
    Multigraphs-2-vectors (mug2vec).

    mug2vec is a sequence of three algorithms that learns a feature vector for each
    input graph.

    Steps:

    1. Pass to ranks - ranks all edge weights from smallest to largest valued edges
    then normalize by a constant.

    2. Omnibus embedding - jointly learns a low dimensional matrix representation for
    all graphs under the random dot product model (RDPG).

    3. Classical MDS (cMDS) - learns a feature vector for each graph by computing
    Euclidean distance between each pair of graph embeddings from omnibus embedding,
    followed by an eigen decomposition.

    Parameters
    ----------
    pass_to_ranks: {'simple-nonzero' (default), 'simple-all', 'zero-boost'} string, or None

        - 'simple-nonzero'
            assigns ranks to all non-zero edges, settling ties using
            the average. Ranks are then scaled by
            :math:`\frac{rank(\text{non-zero edges})}{\text{total non-zero edges} + 1}`
        - 'simple-all'
            assigns ranks to all non-zero edges, settling ties using
            the average. Ranks are then scaled by
            :math:`\frac{rank(\text{non-zero edges})}{n^2 + 1}`
            where n is the number of nodes
        - 'zero-boost'
            preserves the edge weight for all 0s, but ranks the other
            edges as if the ranks of all 0 edges has been assigned. If there are
            10 0-valued edges, the lowest non-zero edge gets weight 11 / (number
            of possible edges). Ties settled by the average of the weight that those
            edges would have received. Number of possible edges is determined
            by the type of graph (loopless or looped, directed or undirected).
        - None
            No pass to ranks applied.

    omnibus_components, cmds_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        ``n_components`` must be ``<= min(X.shape)``. Otherwise, ``n_components`` must be
        ``< min(X.shape)``. If None, then optimal dimensions will be chosen by
        :func:`~graspologic.embed.select_dimension` using ``n_elbows`` argument.

    omnibus_n_elbows, cmds_n_elbows: int, optional, default: 2
        If ``n_components`` is None, then compute the optimal embedding dimension using
        :func:`~graspologic.embed.select_dimension`. Otherwise, ignored.

    Attributes
    ----------
    omnibus_n_components_ : int
        Equals the parameter ``n_components``. If input ``n_components`` was None,
        then equals the optimal embedding dimension.

    cmds_n_components_ : int
        Equals the parameter ``n_components``. If input ``n_components`` was None,
        then equals the optimal embedding dimension.

    embeddings_ : array, shape (n_components, n_features)
        Embeddings from the pipeline. Each graph is a point in ``n_features``
        dimensions.

    See also
    --------
    graspologic.utils.pass_to_ranks
    graspologic.embed.OmnibusEmbed
    graspologic.embed.ClassicalMDS
    graspologic.embed.select_dimension
    """

    def __init__(
        self,
        pass_to_ranks="simple-nonzero",
        omnibus_components=None,
        omnibus_n_elbows=2,
        cmds_components=None,
        cmds_n_elbows=2,
    ):
        self.pass_to_ranks = pass_to_ranks
        self.omnibus_components = omnibus_components
        self.omnibus_n_elbows = omnibus_n_elbows
        self.cmds_components = cmds_components
        self.cmds_n_elbows = cmds_n_elbows

    def _check_inputs(self):
        variables = self.get_params()
        variables.pop("pass_to_ranks")

        for name, val in variables.items():
            if val is not None:
                if not isinstance(val, int):
                    msg = "{} must be an int or None.".format(name)
                    raise ValueError(msg)
                elif val <= 0:
                    msg = "{} must be > 0.".format(name)
                    raise ValueError(msg)

    def fit(self, graphs, y=None):
        """
        Computes a vector for each graph.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).

        y : Ignored

        Returns
        -------
        self : returns an instance of self.
        """
        # Check these prior to PTR just in case
        self._check_inputs()

        if pass_to_ranks is not None:
            graphs = [pass_to_ranks(g, self.pass_to_ranks) for g in graphs]

        omni = OmnibusEmbed(
            n_components=self.omnibus_components, n_elbows=self.omnibus_n_elbows
        )
        omnibus_embedding = omni.fit_transform(graphs)

        self.omnibus_n_components_ = omnibus_embedding.shape[-1]

        cmds = ClassicalMDS(
            n_components=self.cmds_components, n_elbows=self.cmds_n_elbows
        )
        self.embeddings_ = cmds.fit_transform(omnibus_embedding)
        self.cmds_components_ = self.embeddings_.shape[-1]

        return self

    def fit_transform(self, graphs, y=None):
        """
        Computes a vector for each graph.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).

        y : Ignored

        Returns
        -------
        embeddings : returns an instance of self.
        """
        self.fit(graphs)

        return self.embeddings_
