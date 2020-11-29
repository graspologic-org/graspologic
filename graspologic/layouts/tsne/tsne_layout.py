# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import logging
import networkx
import numpy
from sklearn.manifold import TSNE
import time

logger = logging.getLogger(__name__)


def reduce_dimensions(tensors: numpy.array, perplexity: int, n_iters: int):
    start = time.time()
    transformed_points = TSNE(perplexity=perplexity, n_iter=n_iters).fit_transform(
        tensors
    )
    tsne_time = time.time() - start
    logger.info(
        f"tsne completed in {tsne_time} seconds with peplexity: {perplexity} and n_iters: {n_iters}"
    )
    return transformed_points
