# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pytest

from graspy.plot.plot_matrix import matrixplot
from graspy.simulations.simulations import er_np, sbm


def test_matrixplot_inputs():
    X = er_np(100, 0.5)
    meta = pd.DataFrame({
        'hemisphere': np.random.randint(2, size=100),
        'dVNC': np.random.randint(2, size=100),
        'ID': np.random.randint(10, size=100),
    })

    # test matrix
    with pytest.raises(TypeError):
        matrixplot(data="input", meta=meta)
    with pytest.raises(ValueError):
        matrixplot(data=np.zeros((2, 2, 2)), meta=meta)

    # test meta
    with pytest.raises(ValueError):
        matrixplot(X)
    with pytest.raises(ValueError):
        bad_meta = pd.DataFrame({
            'hemisphere': np.random.randint(2, size=1),
            'dVNC': np.random.randint(2, size=1),
            'ID': np.random.randint(10, size=1),
        })
        matrixplot(X, meta=bad_meta)

    # test plot type
    with pytest.raises(ValueError):
        matrixplot(X, plot_type="bad plottype")

    # test sorting_kws
    with pytest.raises(TypeError):
        matrixplot(X, meta=meta, group_class=123)
    with pytest.raises(TypeError):
        matrixplot(X, meta=meta, class_order=123)
    with pytest.raises(TypeError):
        matrixplot(X, meta=meta, item_order=123)
    with pytest.raises(TypeError):
        matrixplot(X, meta=meta, color_class=123)
    with pytest.raises(ValueError):
        matrixplot(X, meta=meta, group_class="bad value")
    with pytest.raises(ValueError):
        matrixplot(X, meta=meta, class_order="bad value")
    with pytest.raises(ValueError):
        matrixplot(X, meta=meta, item_order="bad value")
    with pytest.raises(ValueError):
        matrixplot(X, meta=meta, color_class="bad value")


def test_matrixplot_output():
    """
    simple function to see if plot is made without errors
    """
    X = er_np(10, 0.5)
    meta = pd.DataFrame({
        'hemisphere': np.random.randint(2, size=10),
        'dVNC': np.random.randint(2, size=10),
        'ID': np.random.randint(10, size=10),
    })
    ax = matrixplot(X, meta=meta)
    ax = matrixplot(X, meta=meta, group_class="hemisphere")
    ax = matrixplot(X, meta=meta, group_class="hemisphere", class_order="size")
    ax = matrixplot(X, meta=meta, gropu_class="hemisphere", item_order="ID")
