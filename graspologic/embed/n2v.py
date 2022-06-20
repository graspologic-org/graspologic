# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import logging
import math
import time
from typing import Any, Optional, Union

import networkx as nx
import numpy as np
from gensim.models import Word2Vec

from graspologic.types import List, Tuple
from sklearn.base import BaseEstimator

from ..types import GraphRepresentation
from ..utils import import_graph
from ..pipeline.embed.n2v import *

class Node2VecEmbed(BaseEstimator):

    def __init__(self,
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
        self.labels = None

    def fit_transform(
        self,
        graph: GraphRepresentation,
        *args: Any,
        **kwargs: Any
    ) -> "Node2VecEmbed":
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
        if not isinstance(graph, (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph)):
            graph = import_graph(graph)
            graph = nx.from_numpy_matrix(graph)

        Xemb, labels = node2vec_embed(
            graph,
            num_walks = self.num_walks,
            walk_length = self.walk_length,
            return_hyperparameter = self.return_hyperparameter,
            inout_hyperparameter = self.inout_hyperparameter,
            dimensions = self.dimensions,
            window_size = self.window_size,
            workers = self.workers,
            iterations = self.iterations,
            interpolate_walk_lengths_by_node_degree = self.interpolate_walk_lengths_by_node_degree,
            random_seed = self.random_seed
        )

        self.n_features_in_ = Xemb.shape[0]
        self.is_fitted_ = True
        self.labels = labels

        return Xemb

    def fit(
        self,
        graph: GraphRepresentation,
        *args: Any,
        **kwargs: Any
    ):
        raise NotImplementedError("Use `fit_transform` method instead.")

    def transform(
        self,
        graph: GraphRepresentation,
        *args: Any,
        **kwargs: Any
    ):
        raise NotImplementedError("Use `fit_transform` method instead.")
