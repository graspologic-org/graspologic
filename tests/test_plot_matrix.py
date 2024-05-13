# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np
import pandas as pd
from scipy.sparse import csr_array

from graspologic.plot.plot_matrix import adjplot, matrixplot
from graspologic.simulations.simulations import er_np


class TestPlotMatrix(unittest.TestCase):
    def test_adjplot_inputs(self):
        X = er_np(100, 0.5)
        meta = pd.DataFrame({
            "hemisphere": np.random.randint(2, size=100),
            "region": np.random.randint(2, size=100),
            "cell_size": np.random.randint(10, size=100),
        })

        # test matrix
        with self.assertRaises(TypeError):
            adjplot(data="input", meta=meta)
        with self.assertRaises(ValueError):
            adjplot(data=np.zeros((2, 2, 2)), meta=meta)

        # test meta
        with self.assertRaises(ValueError):
            bad_meta = pd.DataFrame({
                "hemisphere": np.random.randint(2, size=1),
                "region": np.random.randint(2, size=1),
                "cell_size": np.random.randint(10, size=1),
            })
            adjplot(X, meta=bad_meta)

        # test plot type
        with self.assertRaises(ValueError):
            adjplot(X, plot_type="bad plottype")

        # test sorting_kws
        with self.assertRaises(TypeError):
            adjplot(X, meta=meta, group=123)
        with self.assertRaises(TypeError):
            adjplot(X, meta=meta, group_order=123)
        with self.assertRaises(TypeError):
            adjplot(X, meta=meta, item_order=123)
        with self.assertRaises(TypeError):
            adjplot(X, meta=meta, color=123)
        with self.assertRaises(ValueError):
            adjplot(X, meta=meta, group="bad value")
        with self.assertRaises(ValueError):
            adjplot(X, meta=meta, group_order="bad value")
        with self.assertRaises(ValueError):
            adjplot(X, meta=meta, item_order="bad value")
        with self.assertRaises(ValueError):
            adjplot(X, meta=meta, color="bad value")

    def test_adjplot_output(self):
        """
        simple function to see if plot is made without errors
        """
        X = er_np(10, 0.5)
        meta = pd.DataFrame({
            "hemisphere": np.random.randint(2, size=10),
            "region": np.random.randint(2, size=10),
            "cell_size": np.random.randint(10, size=10),
        })
        ax = adjplot(X, meta=meta)
        ax = adjplot(X, meta=meta, group="hemisphere")
        ax = adjplot(X, meta=meta, group="hemisphere", group_order="size")
        ax = adjplot(X, meta=meta, group="hemisphere", item_order="cell_size")

    def test_adjplot_sparse(self):
        X = er_np(10, 0.5)
        adjplot(csr_array(X), plot_type="scattermap")

    def test_matrix_inputs(self):
        X = er_np(100, 0.5)
        meta = pd.DataFrame({
            "hemisphere": np.random.randint(2, size=100),
            "region": np.random.randint(2, size=100),
            "cell_size": np.random.randint(10, size=100),
        })

        # test matrix
        with self.assertRaises(TypeError):
            matrixplot(data="input", col_meta=meta, row_meta=meta)
        with self.assertRaises(ValueError):
            matrixplot(data=np.zeros((2, 2, 2)), col_meta=meta, row_meta=meta)

        # test meta
        with self.assertRaises(ValueError):
            bad_meta = pd.DataFrame({
                "hemisphere": np.random.randint(2, size=1),
                "region": np.random.randint(2, size=1),
                "cell_size": np.random.randint(10, size=1),
            })
            matrixplot(X, col_meta=bad_meta, row_meta=bad_meta)

        # test plot type
        with self.assertRaises(ValueError):
            matrixplot(X, plot_type="bad plottype", col_meta=meta, row_meta=meta)

        # test sorting_kws
        with self.assertRaises(TypeError):
            matrixplot(X, col_meta=meta, row_meta=meta, col_group=123, row_group=123)
        with self.assertRaises(TypeError):
            matrixplot(
                X,
                col_meta=meta,
                col_group_order=123,
                row_meta=meta,
                row_group_order=123,
            )
        with self.assertRaises(TypeError):
            matrixplot(
                X, col_meta=meta, col_item_order=123, row_meta=meta, row_item_order=123
            )
        with self.assertRaises(TypeError):
            matrixplot(X, col_meta=meta, col_color=123, row_meta=meta, row_color=123)
        with self.assertRaises(ValueError):
            matrixplot(
                X,
                col_meta=meta,
                col_group="bad value",
                row_meta=meta,
                row_group="bad value",
            )
        with self.assertRaises(ValueError):
            matrixplot(
                X,
                col_meta=meta,
                col_group_order="bad value",
                row_meta=meta,
                row_group_order="bad value",
            )
        with self.assertRaises(ValueError):
            matrixplot(
                X,
                col_meta=meta,
                col_item_order="bad value",
                row_meta=meta,
                row_item_order="bad value",
            )
        with self.assertRaises(ValueError):
            matrixplot(
                X,
                col_meta=meta,
                col_color="bad value",
                row_meta=meta,
                row_color="bad value",
            )

    def test_matrix_output(self):
        """
        simple function to see if plot is made without errors
        """
        X = er_np(10, 0.5)
        meta = pd.DataFrame({
            "hemisphere": np.random.randint(2, size=10),
            "region": np.random.randint(2, size=10),
            "cell_size": np.random.randint(10, size=10),
        })
        ax = matrixplot(X, col_meta=meta, row_meta=meta)
        ax = matrixplot(X, col_meta=meta, row_meta=meta, row_group="hemisphere")
        ax = matrixplot(
            X,
            col_meta=meta,
            row_meta=meta,
            row_group="hemisphere",
            col_group_order="size",
        )
        ax = matrixplot(
            X,
            col_meta=meta,
            row_meta=meta,
            col_group="hemisphere",
            row_item_order="cell_size",
        )

    def test_matrixplot_sparse(self):
        X = er_np(10, 0.5)
        adjplot(csr_array(X), plot_type="scattermap")
