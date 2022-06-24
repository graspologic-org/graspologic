# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import Any, Optional, Union, List

import numpy as np
import networkx as nx

from sklearn.base import BaseEstimator

from ..types import GraphRepresentation
from ..utils import import_graph
from ..pipeline.embed.n2v_embedding import node2vec_embed


class Node2VecEmbed(BaseEstimator):
    r"""
    Class for computing node2vec embeddings from a given graph. Will follow the word2vec
    algorithm to create the embedding.

    Parameters
    ----------

    graph: Union[nx.Graph, nx.DiGraph]
        A networkx graph or digraph.  A multigraph should be turned into a
        non-multigraph so that the calling user properly handles the multi-edges
        (i.e. aggregate weights or take last edge weight). If the graph is unweighted,
        the weight of each edge will default to 1.
    num_walks : int
        Number of walks per source. Default is 10.
    walk_length: int
        Length of walk per source. Default is 40.
    return_hyperparameter : float
        Return hyperparameter (p). Default is 1.0
    inout_hyperparameter : float
        Inout hyperparameter (q). Default is 1.0
    dimensions : int
        Dimensionality of the word vectors. Default is 128.
    window_size : int
        Maximum distance between the current and predicted word within a sentence.
        Default is 2.
    workers : int
        Use these many worker threads to train the model. Default is 8.
    iterations : int
        Number of epochs in stochastic gradient descent (SGD). Default is 3.
    interpolate_walk_lengths_by_node_degree : bool
        Use a dynamic walk length that corresponds to each nodes
        degree. If the node is in the bottom 20 percentile, default to a walk length of
        1. If it is in the top 10 percentile, use ``walk_length``. If it is in the
        20-80 percentiles, linearly interpolate between 1 and ``walk_length``.
        This will reduce lower degree nodes from biasing your resulting embedding. If a
        low degree node has the same number of walks as a high degree node (which it
        will if this setting is not on), then the lower degree nodes will take a
        smaller breadth of random walks when compared to the high degree nodes. This
        will result in your lower degree walks dominating your higher degree nodes.
    random_seed : int
        Seed to be used for reproducible results. Default is None and will produce a
        random output. Note that for a fully deterministically-reproducible run, you
        must also limit to a single worker thread (`workers=1`), to eliminate ordering
        jitter from OS thread scheduling. In addition the environment variable
        ``PYTHONHASHSEED`` must be set to control hash randomization.

    Returns
    -------
    Tuple[np.array, List[Any]]
        A tuple containing a matrix, with each row index corresponding to the embedding
        for each node. The tuple also contains a vector containing the corresponding
        vertex labels for each row in the matrix. The matrix and vector are
        positionally correlated.

    Notes
    -----
    The original reference implementation of node2vec comes from Aditya Grover from
        https://github.com/aditya-grover/node2vec/.

    Further details on the Alias Method used in this functionality can be found at
        https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    References
    ----------
    .. [1] Aditya Grover and Jure Leskovec  "node2vec: Scalable Feature Learning for
        Networks." Knowledge Discovery and Data Mining, 2016.
    """

    def __init__(
        self,
        num_walks: int = 10,
        walk_length: int = 40,
        return_hyperparameter: float = 1.0,
        inout_hyperparameter: float = 1.0,
        dimensions: int = 128,
        window_size: int = 2,
        workers: int = 8,
        iterations: int = 3,
        interpolate_walk_lengths_by_node_degree: bool = True,
        random_seed: Optional[int] = None,
    ):
        self.num_walks = num_walks
        self.walk_length = walk_length
        self.return_hyperparameter = return_hyperparameter
        self.inout_hyperparameter = inout_hyperparameter
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.iterations = iterations
        self.random_seed = random_seed
        if not isinstance(interpolate_walk_lengths_by_node_degree, bool):
            msg = "Parameter `interpolate_walk_lengths_by_node_degree` is expected to be type bool"
            raise TypeError(msg)
        self.interpolate_walk_lengths_by_node_degree = (
            interpolate_walk_lengths_by_node_degree
        )
        self.labels : List[int] = []

    def fit_transform(
        self, graph: GraphRepresentation, *args: Any, **kwargs: Any
    ) -> np.ndarray:
        """
        Fit Node2Vec model to an input graph.

        Parameters
        ----------
        graph : array-like, scipy.sparse.csr_matrix, or networkx.Graph
            Input graph to embed.

        y: Ignored

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        if not isinstance(
            graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)
        ):
            graph = import_graph(graph)
            graph = nx.from_numpy_matrix(graph)

        Xemb, labels = node2vec_embed(
            graph,
            num_walks=self.num_walks,
            walk_length=self.walk_length,
            return_hyperparameter=self.return_hyperparameter,
            inout_hyperparameter=self.inout_hyperparameter,
            dimensions=self.dimensions,
            window_size=self.window_size,
            workers=self.workers,
            iterations=self.iterations,
            interpolate_walk_lengths_by_node_degree=self.interpolate_walk_lengths_by_node_degree,
            random_seed=self.random_seed,
        )

        self.n_features_in_ = Xemb.shape[0]
        self.is_fitted_ = True
        self.labels = labels

        return Xemb

    def fit(self, graph: GraphRepresentation, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Use `fit_transform` method instead.")

    def transform(self, graph: GraphRepresentation, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError("Use `fit_transform` method instead.")
