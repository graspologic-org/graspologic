# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
import pytest

from graspologic.plot.plot_matrix import adjplot, matrixplot
from graspologic.simulations.simulations import er_np


def test_adjplot_inputs():
    X = er_np(100, 0.5)
    meta = pd.DataFrame(
        {
            "hemisphere": np.random.randint(2, size=100),
            "dVNC": np.random.randint(2, size=100),
            "ID": np.random.randint(10, size=100),
        }
    )

    # test matrix
    with pytest.raises(TypeError):
        adjplot(data="input", meta=meta)
    with pytest.raises(ValueError):
        adjplot(data=np.zeros((2, 2, 2)), meta=meta)

    # test meta
    with pytest.raises(ValueError):
        adjplot(X)
    with pytest.raises(ValueError):
        bad_meta = pd.DataFrame(
            {
                "hemisphere": np.random.randint(2, size=1),
                "dVNC": np.random.randint(2, size=1),
                "ID": np.random.randint(10, size=1),
            }
        )
        adjplot(X, meta=bad_meta)

    # test plot type
    with pytest.raises(ValueError):
        adjplot(X, plot_type="bad plottype")

    # test sorting_kws
    with pytest.raises(TypeError):
        adjplot(X, meta=meta, group_class=123)
    with pytest.raises(TypeError):
        adjplot(X, meta=meta, class_order=123)
    with pytest.raises(TypeError):
        adjplot(X, meta=meta, item_order=123)
    with pytest.raises(TypeError):
        adjplot(X, meta=meta, color_class=123)
    with pytest.raises(ValueError):
        adjplot(X, meta=meta, group_class="bad value")
    with pytest.raises(ValueError):
        adjplot(X, meta=meta, class_order="bad value")
    with pytest.raises(ValueError):
        adjplot(X, meta=meta, item_order="bad value")
    with pytest.raises(ValueError):
        adjplot(X, meta=meta, color_class="bad value")


def test_adjplot_output():
    """
    simple function to see if plot is made without errors
    """
    X = er_np(10, 0.5)
    meta = pd.DataFrame(
        {
            "hemisphere": np.random.randint(2, size=10),
            "dVNC": np.random.randint(2, size=10),
            "ID": np.random.randint(10, size=10),
        }
    )
    ax = adjplot(X, meta=meta)
    ax = adjplot(X, meta=meta, group_class="hemisphere")
    ax = adjplot(X, meta=meta, group_class="hemisphere", class_order="size")
    ax = adjplot(X, meta=meta, group_class="hemisphere", item_order="ID")
