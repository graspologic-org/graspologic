import numpy as np
import pytest

from graspy.plot.plot import heatmap, grid_plot


def test_heatmap_inputs():
    X = np.random.rand(10, 10)
    with pytest.raises(ValueError):
        transform = 'bad transform'
        heatmap(X, transform=transform)