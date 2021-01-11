# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import logging
import math
import time
from typing import Any, List, Optional, Tuple, Union

import networkx as nx
import numpy as np


def node2vec_embed(
    graph: Union[nx.Graph, nx.DiGraph],
    num_walks: int = 10,
    walk_length: int = 80,
    return_hyperparameter: float = 1.0,
    inout_hyperparameter: float = 1.0,
    dimensions: int = 128,
    window_size: int = 10,
    workers: int = 8,
    iterations: int = 1,
    interpolate_walk_lengths_by_node_degree: bool = True,
    random_seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates a node2vec embedding from a given graph. Will follow the word2vec algorithm to create the embedding.

    Parameters
    ----------

    graph: Union[nx.Graph, nx.DiGraph]
        A networkx graph or digraph.  A multigraph should be turned into a non-multigraph so that the calling user
        properly handles the multi-edges (i.e. aggregate weights or take last edge weight).
        If the graph is unweighted, the weight of each edge will default to 1.
    num_walks : int
        Number of walks per source. Default is 10.
    walk_length: int
        Length of walk per source. Default is 80.
    return_hyperparameter : float
        Return hyperparameter (p). Default is 1.0
    inout_hyperparameter : float
        Inout hyperparameter (q). Default is 1.0
    dimensions : int
        Dimensionality of the word vectors. Default is 128.
    window_size : int
        Maximum distance between the current and predicted word within a sentence. Default is 10.
    workers : int
        Use these many worker threads to train the model. Default is 8.
    iterations : int
        Number of epochs in stochastic gradient descent (SGD)
    interpolate_walk_lengths_by_node_degree : bool
        Use a dynamic walk length that corresponds to each nodes
        degree. If the node is in the bottom 20 percentile, default to a walk length of 1. If it is in the top 10
        percentile, use ``walk_length``. If it is in the 20-80 percentiles, linearly interpolate between 1 and ``walk_length``.
        This will reduce lower degree nodes from biasing your resulting embedding. If a low degree node has the same
        number of walks as a high degree node (which it will if this setting is not on), then the lower degree nodes
        will take a smaller breadth of random walks when compared to the high degree nodes. This will result in your
        lower degree walks dominating your higher degree nodes.
    random_seed : int
        Seed to be used for reproducible results. Default is None and will produce a random output. Note that for a fully
        deterministically-reproducible run, you must also limit to a single worker thread (`workers=1`), to eliminate
        ordering jitter from OS thread scheduling. In addition the environment variable ``PYTHONHASHSEED`` must be set
        to control hash randomization.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple containing a matrix, with each row index corresponding to the embedding for each node. The tuple
        also contains a vector containing the corresponding vertex labels for each row in the matrix.
        The matrix and vector are positionally correlated.

    Notes
    -----
    The original reference implementation of node2vec comes from Aditya Grover from
        https://github.com/aditya-grover/node2vec/.

    Further details on the Alias Method used in this functionality can be found at
        https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/

    References
    ----------
    .. [1] Aditya Grover and Jure Leskovec  "node2vec: Scalable Feature Learning for Networks."
        Knowledge Discovery and Data Mining, 2016.
    """

    _preconditions(
        graph,
        num_walks,
        walk_length,
        return_hyperparameter,
        inout_hyperparameter,
        dimensions,
        window_size,
        workers,
        iterations,
        interpolate_walk_lengths_by_node_degree,
    )

    random_state = np.random.RandomState(seed=random_seed)

    node2vec_graph = _Node2VecGraph(
        graph, return_hyperparameter, inout_hyperparameter, random_state
    )

    logging.info(
        f"Starting preprocessing of transition probabilities on graph with {str(len(graph.nodes()))} nodes and "
        f"{str(len(graph.edges()))} edges"
    )

    start = time.time()
    logging.info(f"Starting at time {str(start)}")

    node2vec_graph._preprocess_transition_probabilities()

    logging.info(f"Simulating walks on graph at time {str(time.time())}")
    walks = node2vec_graph._simulate_walks(
        num_walks, walk_length, interpolate_walk_lengths_by_node_degree
    )

    logging.info(f"Learning embeddings at time {str(time.time())}")
    model = _learn_embeddings(
        walks, dimensions, window_size, workers, iterations, random_seed
    )

    end = time.time()
    logging.info(
        f"Completed. Ending time is {str(end)} Elapsed time is {str(start - end)}"
    )

    return model.wv.vectors, model.wv.index2word


def _assert_is_positive_int(name: str, value: int):
    if not isinstance(value, int):
        raise TypeError(f"{name} must be an int")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")


def _assert_is_nonnegative_float(name: str, value: float):
    if not isinstance(value, float):
        raise TypeError(f"{name} must be a float")
    if value < 0.0:
        raise ValueError(f"{name} must be >= 0.0")


def _preconditions(
    graph: Union[nx.Graph, nx.DiGraph],
    num_walks: int,
    walk_length: int,
    return_hyperparameter: float,
    inout_hyperparameter: float,
    dimensions: int,
    window_size: int,
    workers: int,
    iterations: int,
    interpolate_walk_lengths_by_node_degree: bool,
):
    if not isinstance(graph, nx.Graph):
        raise TypeError("graph must be a networkx Graph or DiGraph")
    if graph.is_multigraph():
        raise ValueError(
            "This function does not work on multigraphs - because there are two reasonable ways to treat a "
            "multigraph with different behaviors, we insist that the caller create an appropriate Graph or "
            "DiGraph that represents the manner in which they'd like the multigraph to be treated for the "
            "purposes of this embedding"
        )
    _assert_is_positive_int("num_walks", num_walks)
    _assert_is_positive_int("walk_length", walk_length)
    _assert_is_nonnegative_float("return_hyperparameter", return_hyperparameter)
    _assert_is_nonnegative_float("inout_hyperparameter", inout_hyperparameter)
    _assert_is_positive_int("dimensions", dimensions)
    _assert_is_positive_int("window_size", window_size)
    _assert_is_positive_int("workers", workers)
    _assert_is_positive_int("iterations", iterations)
    if not isinstance(interpolate_walk_lengths_by_node_degree, bool):
        raise TypeError("interpolate_walk_lengths_by_node_degree must be a bool")


def _learn_embeddings(
    walks: List[Any],
    dimensions: int,
    window_size: int,
    workers: int,
    iterations: int,
    random_seed: Optional[int],
):
    """
    Learn embeddings by optimizing the skip-gram objective using SGD.
    """
    from gensim.models import Word2Vec

    walks = [list(map(str, walk)) for walk in walks]

    # Documentation - https://radimrehurek.com/gensim/models/word2vec.html
    model = Word2Vec(
        walks,
        size=dimensions,
        window=window_size,
        min_count=0,
        sg=1,  # Training algorithm: 1 for skip-gram; otherwise CBOW
        workers=workers,
        iter=iterations,
        seed=random_seed,
    )

    return model


class _Node2VecGraph:
    """
    Temporary inner state object for constructing the random walks

    Parameters
    ----------

    graph: nx.Graph
        A networkx graph
    return_hyperparameter : float
        Return hyperparameter
    inout_hyperparameter : float
        Inout hyperparameter
    random_state : np.random.RandomState
        Random State for reproducible results. Default is None and will produce random
        results
    """

    def __init__(
        self,
        graph: nx.Graph,
        return_hyperparameter: float,
        inout_hyperparameter: float,
        random_state: Optional[np.random.RandomState] = None,
    ):
        self.graph: nx.Graph = graph
        self.is_directed = self.graph.is_directed()
        self.p = return_hyperparameter
        self.q = inout_hyperparameter
        self.random_state = random_state

    def node2vec_walk(
        self,
        walk_length: int,
        start_node: Any,
        degree_percentiles: Optional[np.ndarray],
    ):
        """
        Simulate a random walk starting from start node.
        """
        graph = self.graph
        alias_nodes = self.alias_nodes
        alias_edges = self.alias_edges

        walk = [start_node]

        # Percentiles will be provided if we are using the 'interpolate_walk_lengths_by_node_degree' feature.
        # the intent of the code is to default the bottom 20% of to a minimal walk length, default the top 10% to a
        # maximum walk length, and interpolate the inner 70% linearly from min to max.

        # This is to avoid having your random walks be dominated by low degree nodes. If the low degree nodes have the
        # same number of walks as the high degree nodes, the low degree nodes will take a smaller breadth of paths
        # (due to their being less nodes to choose from) and will bias your resulting Word2Vec embedding
        if degree_percentiles is not None:
            degree = nx.degree(graph, start_node)
            walk_length = self._get_walk_length_interpolated(
                degree, degree_percentiles, walk_length
            )

        while len(walk) < walk_length:
            current = walk[-1]
            current_neighbors = sorted(graph.neighbors(current))

            if len(current_neighbors) > 0:
                if len(walk) == 1:
                    walk.append(
                        current_neighbors[
                            _alias_draw(
                                alias_nodes[current][0],
                                alias_nodes[current][1],
                                self.random_state,
                            )
                        ]
                    )
                else:
                    prev = walk[-2]
                    next = current_neighbors[
                        _alias_draw(
                            alias_edges[(prev, current)][0],
                            alias_edges[(prev, current)][1],
                            self.random_state,
                        )
                    ]
                    walk.append(next)
            else:
                break

        return walk

    @staticmethod
    def _get_walk_length_interpolated(
        degree: int, percentiles: np.ndarray, max_walk_length: int
    ):
        """
        Given a node's degree, determine the length of a walk that should be used. If the degree is less than the
        first element of the percentiles list, default the walk length to 1. Otherwise, if the degree is greater
        than the last element of the list, default it to the max_walk_length. If it falls in the middle, do a linear
        interpolation to decide the length of the walk.
        """

        new_walk_length = None

        for i, percentile in enumerate(percentiles):
            # if we are below the first percentile in the list, default to a walk length of 1
            if i == 0 and degree < percentile:
                return 1

            # otherwise, find which bucket we are going to be in.
            if degree <= percentile:
                new_walk_length = max_walk_length * ((i * 0.1) + 0.2)
                break

        # the degree is above the last percentile
        if not new_walk_length:
            new_walk_length = max_walk_length

        # a walk length of 0 is invalid but can happen depending on the percentiles used
        if new_walk_length < 1:
            new_walk_length = 1

        return math.floor(new_walk_length)

    def _simulate_walks(
        self,
        num_walks: int,
        walk_length: int,
        interpolate_walk_lengths_by_node_degree: bool = False,
    ):
        """
        Repeatedly simulate random walks from each node.
        """
        graph = self.graph
        walks = []
        nodes = list(graph.nodes())

        degree_percentiles: Optional[np.ndarray] = None
        if interpolate_walk_lengths_by_node_degree:
            degree_percentiles = np.percentile(
                [degree for _, degree in graph.degree()], [x for x in range(20, 90, 10)]
            )

        for walk_iteration in range(num_walks):
            logging.info(
                "Walk iteration: " + str(walk_iteration + 1) + "/" + str(num_walks)
            )

            self.random_state.shuffle(nodes)
            for node in nodes:
                walks.append(
                    self.node2vec_walk(
                        walk_length=walk_length,
                        start_node=node,
                        degree_percentiles=degree_percentiles,
                    )
                )

        return walks

    def _get_alias_edge(self, source: Any, destination: Any):
        """
        Get the alias edge setup lists for a given edge.
        """
        graph = self.graph
        p = self.p
        q = self.q

        unnormalized_probs = []
        for destination_neighbor in sorted(graph.neighbors(destination)):
            if destination_neighbor == source:
                unnormalized_probs.append(
                    graph[destination][destination_neighbor].get("weight", 1) / p
                )
            elif graph.has_edge(destination_neighbor, source):
                unnormalized_probs.append(
                    graph[destination][destination_neighbor].get("weight", 1)
                )
            else:
                unnormalized_probs.append(
                    graph[destination][destination_neighbor].get("weight", 1) / q
                )
        norm_const = sum(unnormalized_probs)
        normalized_probs = [float(u_prob) / norm_const for u_prob in unnormalized_probs]

        return _alias_setup(normalized_probs)

    def _preprocess_transition_probabilities(self, weight_default: float = 1.0):
        """
        Preprocessing of transition probabilities for guiding the random walks.
        """
        graph = self.graph
        is_directed = self.is_directed

        alias_nodes = {}
        total_nodes = len(graph.nodes())
        bucket = 0
        current_node = 0
        quotient = int(total_nodes / 10)

        logging.info(
            f"Beginning preprocessing of transition probabilities for {total_nodes} vertices"
        )
        for node in graph.nodes():
            current_node += 1
            if current_node > bucket * quotient:
                bucket += 1
                logging.info(f"Completed {current_node} / {total_nodes} vertices")

            unnormalized_probs = [
                graph[node][nbr].get("weight", weight_default)
                for nbr in sorted(graph.neighbors(node))
            ]
            norm_const = sum(unnormalized_probs)
            normalized_probs = [
                float(u_prob) / norm_const for u_prob in unnormalized_probs
            ]
            alias_nodes[node] = _alias_setup(normalized_probs)
        logging.info(
            f"Completed preprocessing of transition probabilities for vertices"
        )

        alias_edges = {}

        total_edges = len(graph.edges())
        bucket = 0
        current_edge = 0
        quotient = int(total_edges / 10)

        logging.info(
            f"Beginning preprocessing of transition probabilities for {total_edges} edges"
        )
        if is_directed:
            for edge in graph.edges():
                current_edge += 1
                if current_edge > bucket * quotient:
                    bucket += 1
                    logging.info(f"Completed {current_edge} / {total_edges} edges")

                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
        else:
            for edge in graph.edges():
                current_edge += 1
                if current_edge > bucket * quotient:
                    bucket += 1
                    logging.info(f"Completed {current_edge} / {total_edges} edges")

                alias_edges[edge] = self._get_alias_edge(edge[0], edge[1])
                alias_edges[(edge[1], edge[0])] = self._get_alias_edge(edge[1], edge[0])

        logging.info(f"Completed preprocessing of transition probabilities for edges")

        self.alias_nodes = alias_nodes
        self.alias_edges = alias_edges

        return


def _alias_setup(probabilities: List[float]):
    """
    Compute utility lists for non-uniform sampling from discrete distributions.
    Refer to
     https://lips.cs.princeton.edu/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
    for details
    """
    number_of_outcomes = len(probabilities)
    alias = np.zeros(number_of_outcomes)
    sampled_probabilities = np.zeros(number_of_outcomes, dtype=np.int)

    smaller = []
    larger = []
    for i, prob in enumerate(probabilities):
        alias[i] = number_of_outcomes * prob
        if alias[i] < 1.0:
            smaller.append(i)
        else:
            larger.append(i)

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()
        large = larger.pop()

        sampled_probabilities[small] = large
        alias[large] = alias[large] + alias[small] - 1.0
        if alias[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return sampled_probabilities, alias


def _alias_draw(
    probabilities: List[float], alias: List[float], random_state: np.random.RandomState
):
    """
    Draw sample from a non-uniform discrete distribution using alias sampling.
    """
    number_of_outcomes = len(probabilities)
    random_index = int(np.floor(random_state.rand() * number_of_outcomes))

    if random_state.rand() < alias[random_index]:
        return random_index
    else:
        return probabilities[random_index]
