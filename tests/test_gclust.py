import pytest
import numpy as np
from numpy import array_equal, allclose

from graspy.cluster import GaussianCluster


def test_inputs():
    # Generate random data
    X = np.random.normal(0, 1, size=(10, 2))

    # max_cluster > n_samples
    with pytest.raises(ValueError):
        gclust = GaussianCluster(100)
        gclust.fit(X)

    # max_cluster < 0
    with pytest.raises(ValueError):
        gclust = GaussianCluster(max_components=-1)
        gclust.fit(X)
