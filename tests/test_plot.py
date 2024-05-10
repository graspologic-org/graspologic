# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import beartype.roar
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_array
from sklearn.mixture import GaussianMixture

from graspologic.plot.plot import (
    _sort_inds,
    gridplot,
    heatmap,
    networkplot,
    pairplot,
    pairplot_with_gmm,
)
from graspologic.simulations.simulations import er_np, sbm


def _test_pairplot_with_gmm_inputs(caller: unittest.TestCase, **kws):
    X = np.random.rand(15, 3)
    gmm = GaussianMixture(n_components=3, **kws).fit(X)
    labels = ["A"] * 5 + ["B"] * 5 + ["C"] * 5
    # test data
    with caller.assertRaises(ValueError):
        pairplot_with_gmm(X="test", gmm=gmm)

    with caller.assertRaises(ValueError):
        pairplot_with_gmm(X=X, gmm=gmm, labels=["A"])

    with caller.assertRaises(NameError):
        pairplot_with_gmm(X, gmm=None)


def _test_pairplot_with_gmm_outputs(**kws):
    X = np.random.rand(15, 3)
    gmm = GaussianMixture(n_components=3, **kws).fit(X)
    labels = ["A"] * 5 + ["B"] * 5 + ["C"] * 5
    cluster_palette = {0: "red", 1: "blue", 2: "green"}
    label_palette = {"A": "red", "B": "blue", "C": "green"}
    fig = pairplot_with_gmm(X, gmm)
    fig = pairplot_with_gmm(
        X,
        gmm,
        labels=labels,
        cluster_palette=cluster_palette,
        label_palette=label_palette,
    )


class TestPlot(unittest.TestCase):
    def test_common_inputs(self):
        X = er_np(100, 0.5)
        x = np.random.rand(100, 1)
        y = np.random.rand(100, 1)
        grid_labels = ["Test1"]

        # test figsize
        figsize = "bad figsize"
        with self.assertRaises(TypeError):
            heatmap(X, figsize=figsize)
        with self.assertRaises(beartype.roar.BeartypeCallHintParamViolation):
            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, figsize=figsize)

        # test height
        height = "1"
        with self.assertRaises(TypeError):
            gridplot([X], grid_labels, height=height)
        with self.assertRaises(TypeError):
            pairplot(X, height=height)

        # test title
        title = 1
        with self.assertRaises(TypeError):
            heatmap(X, title=title)
        with self.assertRaises(TypeError):
            gridplot([X], grid_labels, title=title)
        with self.assertRaises(TypeError):
            pairplot(X, title=title)
        with self.assertRaises(beartype.roar.BeartypeCallHintParamViolation):
            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, title=title)

        # test context
        context = 123
        with self.assertRaises(TypeError):
            heatmap(X, context=context)
        with self.assertRaises(TypeError):
            gridplot([X], grid_labels, context=context)
        with self.assertRaises(TypeError):
            pairplot(X, context=context)
        with self.assertRaises(beartype.roar.BeartypeCallHintParamViolation):
            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, context=context)

        context = "journal"
        with self.assertRaises(ValueError):
            heatmap(X, context=context)
        with self.assertRaises(ValueError):
            gridplot([X], grid_labels, context=context)
        with self.assertRaises(ValueError):
            pairplot(X, context=context)
        with self.assertRaises(ValueError):
            networkplot(adjacency=X, x=x, y=y, context=context)

        # test font scales
        font_scales = ["1", []]
        for font_scale in font_scales:
            with self.assertRaises(TypeError):
                heatmap(X, font_scale=font_scale)
            with self.assertRaises(TypeError):
                gridplot([X], grid_labels, font_scale=font_scale)
            with self.assertRaises(TypeError):
                pairplot(X, font_scale=font_scale)
            with self.assertRaises(beartype.roar.BeartypeCallHintParamViolation):
                with self.assertRaises(TypeError):
                    networkplot(adjacency=X, x=x, y=y, font_scale=font_scale)

        # ticklabels
        with self.assertRaises(TypeError):
            xticklabels = "labels"
            yticklabels = "labels"
            heatmap(X, xticklabels=xticklabels, yticklabels=yticklabels)

        with self.assertRaises(ValueError):
            xticklabels = ["{}".format(i) for i in range(5)]
            yticklabels = ["{}".format(i) for i in range(5)]
            heatmap(X, xticklabels=xticklabels, yticklabels=yticklabels)

        with self.assertRaises(TypeError):
            heatmap(X, title_pad="f")

        with self.assertRaises(TypeError):
            gridplot([X], title_pad="f")

        with self.assertRaises(TypeError):
            heatmap(X, hier_label_fontsize="f")

        with self.assertRaises(TypeError):
            gridplot([X], hier_label_fontsize="f")

    def test_heatmap_inputs(self):
        """
        test parameter checks
        """
        X = np.random.rand(10, 10)

        with self.assertRaises(TypeError):
            heatmap(X="input")

        # transform
        with self.assertRaises(ValueError):
            transform = "bad transform"
            heatmap(X, transform=transform)

        # cmap
        with self.assertRaises(TypeError):
            cmap = 123
            heatmap(X, cmap=cmap)

        # center
        with self.assertRaises(TypeError):
            center = "center"
            heatmap(X, center=center)

        # cbar
        with self.assertRaises(TypeError):
            cbar = 1
            heatmap(X, cbar=cbar)

    def test_heatmap_output(self):
        """
        simple function to see if plot is made without errors
        """
        X = er_np(10, 0.5)
        xticklabels = ["Dimension {}".format(i) for i in range(10)]
        yticklabels = ["Dimension {}".format(i) for i in range(10)]

        fig = heatmap(
            X, transform="log", xticklabels=xticklabels, yticklabels=yticklabels
        )
        fig = heatmap(X, transform="zero-boost")
        fig = heatmap(X, transform="simple-all")
        fig = heatmap(X, transform="simple-nonzero")
        fig = heatmap(X, transform="binarize")
        fig = heatmap(X, cmap="gist_rainbow")

    def test_gridplot_inputs(self):
        X = [er_np(10, 0.5)]
        labels = ["ER(10, 0.5)"]

        with self.assertRaises(TypeError):
            gridplot(X="input", labels=labels)

        with self.assertRaises(ValueError):
            gridplot(X, labels=["a", "b"])

        # transform
        with self.assertRaises(ValueError):
            transform = "bad transform"
            gridplot(X, labels=labels, transform=transform)

    def test_gridplot_outputs(self):
        """
        simple function to see if plot is made without errors
        """
        X = [er_np(10, 0.5) for _ in range(2)]
        labels = ["Random A", "Random B"]
        fig = gridplot(X, labels)
        fig = gridplot(X, labels, transform="zero-boost")
        fig = gridplot(X, labels, "simple-all", title="Test", font_scale=0.9)

    def test_pairplot_inputs(self):
        X = np.random.rand(15, 3)
        Y = ["A"] * 5 + ["B"] * 5 + ["C"] * 5

        # test data
        with self.assertRaises(TypeError):
            pairplot(X="test")

        with self.assertRaises(ValueError):
            pairplot(X=X, labels=["A"])

        with self.assertRaises(TypeError):
            pairplot(X, col_names="A")

        with self.assertRaises(ValueError):
            pairplot(X, col_names=["1", "2"])

        with self.assertRaises(ValueError):
            pairplot(X, col_names=["1", "2", "3"], variables=[1, 2, 3, 4])

        with self.assertRaises(KeyError):
            pairplot(X, col_names=["1", "2", "3"], variables=["A", "B"])

    def test_pairplot_outputs(self):
        X = np.random.rand(15, 3)
        Y = ["A"] * 5 + ["B"] * 5 + ["C"] * 5
        col_names = ["Feature1", "Feature2", "Feature3"]

        fig = pairplot(
            X,
            Y,
            col_names,
            title="Test",
            height=1.5,
            variables=["Feature1", "Feature2"],
        )

    def test_pairplot_with_gmm_inputs_type_full(self):
        _test_pairplot_with_gmm_inputs(self, covariance_type="full")

    def test_pairplot_with_gmm_inputs_type_diag(self):
        _test_pairplot_with_gmm_inputs(self, covariance_type="diag")

    def test_pairplot_with_gmm_inputs_type_tied(self):
        _test_pairplot_with_gmm_inputs(self, covariance_type="tied")

    def test_pairplot_with_gmm_inputs_type_spherical(self):
        _test_pairplot_with_gmm_inputs(self, covariance_type="spherical")

    def test_pairplot_with_gmm_outputs_type_full(self):
        _test_pairplot_with_gmm_outputs(covariance_type="full")

    def test_pairplot_with_gmm_outputs_type_diag(self):
        _test_pairplot_with_gmm_outputs(covariance_type="diag")

    def test_pairplot_with_gmm_outputs_type_tied(self):
        _test_pairplot_with_gmm_outputs(covariance_type="tied")

    def test_pairplot_with_gmm_outputs_type_spherical(self):
        _test_pairplot_with_gmm_outputs(covariance_type="spherical")

    def test_networkplot_inputs(self):
        X = er_np(15, 0.5)
        x = np.random.rand(15, 1)
        y = np.random.rand(15, 1)
        with self.assertRaises(beartype.roar.BeartypeCallHintParamViolation):
            with self.assertRaises(TypeError):
                networkplot(adjacency="test", x=x, y=y)

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=["A"], y=["A"])

            with self.assertRaises(TypeError):
                networkplot(
                    adjacency=csr_array(X), x="source", y="target", node_data="data"
                )

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, node_data="data")

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, node_hue=(5, 5))

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, palette=4)

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, node_size=(5, 5))

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, node_sizes=4)

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, node_alpha="test")

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, edge_hue=4)

            with self.assertRaises(TypeError):
                networkplot(adjacency=csr_array(X), x=x, y=y, edge_alpha="test")

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, edge_linewidth="test")

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, ax="test")

            with self.assertRaises(TypeError):
                networkplot(adjacency=X, x=x, y=y, legend=4)

    def test_networkplot_outputs_int(self):
        X = er_np(15, 0.5)
        xarray = np.random.rand(15, 1)
        yarray = np.random.rand(15, 1)
        xstring = "source"
        ystring = "target"
        node_df = pd.DataFrame(index=range(X.shape[0]))
        node_df.loc[:, "source"] = xarray
        node_df.loc[:, "target"] = yarray
        hue = np.random.randint(2, size=15)
        palette = {0: (0.8, 0.4, 0.2), 1: (0, 0.9, 0.4)}
        size = np.random.rand(15)
        sizes = (10, 200)

        fig = networkplot(adjacency=X, x=xarray, y=yarray)
        fig = networkplot(adjacency=csr_array(X), x=xarray, y=yarray)
        fig = networkplot(adjacency=X, x=xstring, y=ystring, node_data=node_df)
        fig = networkplot(
            adjacency=csr_array(X), x=xstring, y=ystring, node_data=node_df
        )
        fig = plt.figure()
        ax = fig.add_subplot(211)
        fig = networkplot(
            adjacency=X,
            x=xarray,
            y=yarray,
            node_hue=hue,
            palette=palette,
            node_size=size,
            node_sizes=sizes,
            node_alpha=0.5,
            edge_alpha=0.4,
            edge_linewidth=0.6,
            ax=ax,
        )

    def test_networkplot_outputs_str(self):
        X = er_np(15, 0.7)
        node_df = pd.DataFrame(index=["node {}".format(i) for i in range(15)])
        node_df.loc[:, "source"] = np.random.rand(15, 1)
        node_df.loc[:, "target"] = np.random.rand(15, 1)
        node_df.loc[:, "hue"] = np.random.randint(2, size=15)
        palette = {0: (0.8, 0.4, 0.2), 1: (0, 0.9, 0.4)}
        size = np.random.rand(15)
        sizes = (10, 200)

        fig = networkplot(
            adjacency=X,
            x="source",
            y="target",
            node_data=node_df,
            node_hue="hue",
            palette=palette,
            node_size=size,
            node_sizes=sizes,
            node_alpha=0.5,
            edge_alpha=0.4,
            edge_linewidth=0.6,
        )

    def test_sort_inds(self):
        B = np.array([
            [0, 0.2, 0.1, 0.1, 0.1],
            [0.2, 0.8, 0.1, 0.3, 0.1],
            [0.15, 0.1, 0, 0.05, 0.1],
            [0.1, 0.1, 0.2, 1, 0.1],
            [0.1, 0.2, 0.1, 0.1, 0.8],
        ])

        g = sbm([10, 30, 50, 25, 25], B, directed=True)
        degrees = g.sum(axis=0) + g.sum(axis=1)
        degree_sort_inds = np.argsort(degrees)
        labels2 = 40 * ["0"] + 100 * ["1"]
        labels1 = 10 * ["d"] + 30 * ["c"] + 50 * ["d"] + 25 * ["e"] + 25 * ["c"]
        labels1 = np.array(labels1)
        labels2 = np.array(labels2)
        sorted_inds = _sort_inds(g, labels1, labels2, True)
        # sort outer blocks first if given, sort by num verts in the block
        # for inner hier, sort by num verts for that category across the entire graph
        # ie if there are multiple inner hier across different outer blocks, sort
        # by prevalence in the entire graph, not within block
        # this is to make the ordering within outer block consistent
        # within a block, sort by degree

        # outer block order should thus be: 1, 0
        # inner block order should thus be: d, c, e

        # show that outer blocks are sorted correctly
        labels2 = labels2[sorted_inds]
        self.assertTrue(np.all(labels2[:100] == "1"))
        self.assertTrue(np.all(labels2[100:] == "0"))

        # show that inner blocks are sorted correctly
        labels1 = labels1[sorted_inds]
        self.assertTrue(np.all(labels1[:50] == "d"))
        self.assertTrue(np.all(labels1[50:75] == "c"))
        self.assertTrue(np.all(labels1[75:100] == "e"))
        self.assertTrue(np.all(labels1[100:110] == "d"))
        self.assertTrue(np.all(labels1[110:] == "c"))

        # show that within block, everything is in descending degree order
        degrees = degrees[sorted_inds]
        self.assertTrue(np.all(np.diff(degrees[:50]) <= 0))
        self.assertTrue(np.all(np.diff(degrees[50:75]) <= 0))
        self.assertTrue(np.all(np.diff(degrees[75:100]) <= 0))
        self.assertTrue(np.all(np.diff(degrees[100:110]) <= 0))
        self.assertTrue(np.all(np.diff(degrees[110:]) <= 0))
