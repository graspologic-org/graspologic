# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pytest
from numpy.testing import assert_equal

from graspologic.cluster import GaussianCluster
from graspologic.embed import mug2vec
from graspologic.simulations import sbm


def generate_data():
    np.random.seed(1)

    p1 = [[0.2, 0.1], [0.1, 0.2]]
    p2 = [[0.1, 0.2], [0.2, 0.1]]
    n = [50, 50]

    g1 = [sbm(n, p1) for _ in range(20)]
    g2 = [sbm(n, p2) for _ in range(20)]
    g = g1 + g2

    y = ["0"] * 20 + ["1"] * 20

    return g, y


def test_mug2vec():
    graphs, labels = generate_data()

    mugs = mug2vec(pass_to_ranks=None)
    xhat = mugs.fit_transform(graphs)

    gmm = GaussianCluster(5)
    gmm.fit(xhat, labels)

    assert_equal(gmm.n_components_, 2)


def test_inputs():
    graphs, labels = generate_data()

    mugs = mug2vec(omnibus_components=-1)
    with pytest.raises(ValueError):
        mugs.fit(graphs)

    mugs = mug2vec(cmds_components=-1)
    with pytest.raises(ValueError):
        mugs.fit(graphs)

    mugs = mug2vec(omnibus_n_elbows=-1)
    with pytest.raises(ValueError):
        mugs.fit(graphs)

    mugs = mug2vec(cmds_n_elbows=-1)
    with pytest.raises(ValueError):
        mugs.fit(graphs)
