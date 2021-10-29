# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np
from sklearn.mixture import GaussianMixture

from graspologic.plot.plot import (
    _sort_inds,
    gridplot,
    heatmap,
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
        grid_labels = ["Test1"]

        # test figsize
        with self.assertRaises(TypeError):
            figsize = "bad figsize"
            heatmap(X, figsize=figsize)

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

        # test context
        context = 123
        with self.assertRaises(TypeError):
            heatmap(X, context=context)
        with self.assertRaises(TypeError):
            gridplot([X], grid_labels, context=context)
        with self.assertRaises(TypeError):
            pairplot(X, context=context)

        context = "journal"
        with self.assertRaises(ValueError):
            heatmap(X, context=context)
        with self.assertRaises(ValueError):
            gridplot([X], grid_labels, context=context)
        with self.assertRaises(ValueError):
            pairplot(X, context=context)

        # test font scales
        font_scales = ["1", []]
        for font_scale in font_scales:
            with self.assertRaises(TypeError):
                heatmap(X, font_scale=font_scale)
            with self.assertRaises(TypeError):
                gridplot([X], grid_labels, font_scale=font_scale)
            with self.assertRaises(TypeError):
                pairplot(X, cont_scale=font_scale)

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

        fig = pairplot(X)
        fig = pairplot(X, Y)
        fig = pairplot(X, Y, col_names)
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

    def test_sort_inds(self):
        B = np.array(
            [
                [0, 0.2, 0.1, 0.1, 0.1],
                [0.2, 0.8, 0.1, 0.3, 0.1],
                [0.15, 0.1, 0, 0.05, 0.1],
                [0.1, 0.1, 0.2, 1, 0.1],
                [0.1, 0.2, 0.1, 0.1, 0.8],
            ]
        )

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
