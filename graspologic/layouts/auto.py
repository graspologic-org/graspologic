# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from typing import Any, Dict, List, Tuple

import logging
import math
import time

import networkx as nx
import numpy as np

from sklearn.manifold import TSNE
import umap

from .classes import NodePosition
from ..embed import node2vec_embed
from ..partition import leiden
from ..preprocessing import cut_edges_by_weight, histogram_edge_weight

logger = logging.getLogger(__name__)


# automatically generates a layout by running node2vec over the graph and down
# projecting the embedding into 2d space via umap or tsne
def layout_tsne(
    graph: nx.Graph,
    perplexity: int,
    n_iter: int,
    max_edges: int = 10000000,
) -> Tuple[nx.Graph, List[NodePosition]]:
    lcc_graph, tensors, labels = _node2vec_for_layout(graph, max_edges)
    points = TSNE(perplexity=perplexity, n_iter=n_iter).fit_transform(tensors)
    positions = _node_positions_from(lcc_graph, labels, points)
    return lcc_graph, positions


def layout_umap(
    graph: nx.Graph,
    min_dist: float = 0.75,
    n_neighbors: int = 25,
    max_edges: int = 10000000,
) -> Tuple[nx.Graph, List[NodePosition]]:
    lcc_graph, tensors, labels = _node2vec_for_layout(graph, max_edges)
    points = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors).fit_transform(
        tensors
    )
    positions = _node_positions_from(lcc_graph, labels, points)
    return lcc_graph, positions


def _largest_connected_component(graph: nx.Graph) -> nx.Graph:
    largest_component = max(nx.connected_components(graph), key=len)
    return graph.subgraph(largest_component).copy()


def _approximate_prune(graph: nx.Graph, max_edges_to_keep: int = 1000000):
    num_edges = len(graph.edges())
    logger.info(f"num edges: {num_edges}")

    if num_edges > max_edges_to_keep:
        histogram, bins = histogram_edge_weight(graph, bin_directive=100)
        counts = 0
        bin_edge_for_maximum_weight = bins[0]
        for i, count in reversed(list(enumerate(histogram))):
            counts += count
            if counts >= max_edges_to_keep:
                bin_edge_for_maximum_weight = bins[i + 1]
                break
        threshold = bins[bin_edge_for_maximum_weight]
        graph = cut_edges_by_weight(
            graph, cut_threshold=threshold, cut_process="smaller_than_inclusive"
        )
        logger.debug(f"after cut num edges: {len(graph.edges())}")

    return graph


def _node2vec_for_layout(
    graph: nx.Graph,
    max_edges: int = 10000000,
) -> Tuple[nx.Graph, np.ndarray, np.ndarray]:
    graph = _approximate_prune(graph, max_edges)
    graph = _largest_connected_component(graph)

    start = time.time()
    tensors, labels = node2vec_embed(
        graph=graph, dimensions=128, num_walks=10, window_size=2, iterations=3
    )
    embedding_time = time.time() - start
    logger.info(f"embedding completed in {embedding_time} seconds")
    return graph, tensors, labels


def _node_positions_from(
    graph: nx.Graph,
    labels: np.ndarray,
    down_projection_2d: np.ndarray,
) -> List[NodePosition]:
    degree = graph.degree()
    sizes = _compute_sizes(degree)
    covered_area = _covered_size(sizes)
    scaled_points = _scale_points(down_projection_2d, covered_area)
    partitions = leiden(graph)
    positions = [
        NodePosition(
            node_id=key,
            x=scaled_points[index][0],
            y=scaled_points[index][1],
            size=sizes[key],
            community=partitions[key],
        )
        for index, key in enumerate(labels)
    ]
    return positions


def _find_min_max_degree(
    degrees: nx.classes.reportviews.DegreeView,
) -> Tuple[float, float]:
    min_degree = math.inf
    max_degree = -math.inf
    for _, degree in degrees:
        min_degree = min(min_degree, degree)
        max_degree = max(max_degree, degree)
    return min_degree, max_degree


def _compute_sizes(
    degrees: nx.classes.reportviews.DegreeView,
    min_size: float = 5.0,
    max_size: float = 150.0,
) -> Dict[Any, float]:
    min_degree, max_degree = _find_min_max_degree(degrees)
    sizes = {}
    for node_id, degree in degrees:
        if max_degree == min_degree:
            size = min_size
        else:
            normalized = (degree - min_degree) / (max_degree - min_degree)
            size = normalized * (max_size - min_size) + min_size
        sizes[node_id] = size
    return sizes


def _covered_size(sizes: Dict[Any, float]) -> float:
    total = sum(math.pow(value, 2) * math.pi for value in sizes.values())
    return total


def _get_bounds(points: np.ndarray) -> Tuple[float, float, float, float]:
    min_x, min_y, max_x, max_y = math.inf, math.inf, -math.inf, -math.inf
    for x, y in points:
        min_x = min(x, min_x)
        min_y = min(y, min_y)
        max_x = max(x, max_x)
        max_y = max(y, max_y)
    return min_x, min_y, max_x, max_y


def _center_points_on_origin(points: np.ndarray) -> np.ndarray:
    min_x, min_y, max_x, max_y = _get_bounds(points)
    x_range = max_x - min_x
    y_range = max_y - min_y
    center_x = x_range / 2 + min_x
    center_y = y_range / 2 + min_y
    moved_points = [(x - center_x, y - center_y) for x, y in points]
    return np.array(moved_points)


def _scale_to_unit_square(points: np.ndarray) -> np.ndarray:
    _, _, max_x, max_y = _get_bounds(points)
    points[:, 0] /= max_x
    points[:, 1] /= max_y
    return points


def _scale_to_new_bounds(points: np.ndarray, max_x: float, max_y: float) -> np.ndarray:
    points[:, 0] *= max_x
    points[:, 1] *= max_y
    return points


def _new_bounds(
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
    covered_area: float,
    target_ratio: float,
) -> Tuple[float, float, float, float]:
    range_x = max_x - min_x
    mid_x = min_x + range_x / 2
    range_y = max_y - min_y
    mid_y = min_y + range_y / 2
    range_ratio = range_x / range_y

    new_area = covered_area / target_ratio
    new_range_y = math.sqrt(new_area / range_ratio)
    new_range_x = new_area / new_range_y
    new_min_x, new_min_y, new_max_x, new_max_y = (
        mid_x - new_range_x / 2,
        mid_y - new_range_y / 2,
        mid_x + new_range_x / 2,
        mid_y + new_range_y / 2,
    )

    return new_min_x, new_min_y, new_max_x, new_max_y


def _scale_points(
    points: np.ndarray, covered_area: float, target_ratio: float = 0.14
) -> np.ndarray:
    # through lots of experiments we have found 14% to be a good ratio
    # of color to whitespace
    moved_points = _center_points_on_origin(points)
    min_x, min_y, max_x, max_y = _get_bounds(moved_points)

    _, _, new_max_x, new_max_y = _new_bounds(
        min_x, min_y, max_x, max_y, covered_area, target_ratio
    )

    scaled_points = _scale_to_unit_square(moved_points)

    final_locs = _scale_to_new_bounds(scaled_points, new_max_x, new_max_y)

    return final_locs
