# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from numpy.random import normal, poisson
from numpy.testing import assert_equal
from scipy.sparse import csr_array
from sklearn.base import clone
from sklearn.metrics import adjusted_rand_score, pairwise_distances
from sklearn.mixture import GaussianMixture

from graspologic.embed.ase import AdjacencySpectralEmbed
from graspologic.embed.lse import LaplacianSpectralEmbed
from graspologic.simulations.simulations import er_nm, er_np, sbm
from graspologic.utils import remove_vertices


def _test_output_dim_directed(self, method):
    n_components = 4
    embed = method(n_components=n_components, concat=True)
    n = 10
    M = 20
    A = er_nm(n, M, directed=True) + 5
    self.assertEqual(embed.fit_transform(A).shape, (n, 8))
    self.assertEqual(embed.latent_left_.shape, (n, 4))
    self.assertEqual(embed.latent_right_.shape, (n, 4))


def _test_output_dim(self, method, sparse=False, *args, **kwargs):
    n_components = 4
    embed = method(n_components=n_components)
    n = 10
    M = 20
    A = er_nm(n, M) + 5
    if sparse:
        A = csr_array(A)
    embed._reduce_dim(A)
    self.assertEqual(embed.latent_left_.shape, (n, 4))
    self.assertTrue(embed.latent_right_ is None)


def _test_sbm_er_binary(self, method, P, directed=False, sparse=False, *args, **kwargs):
    verts = 100
    communities = 2

    verts_per_community = [50, 50]

    sbm_sample, labels_sbm = sbm(
        verts_per_community, P, directed=directed, return_labels=True
    )

    if sparse:
        sbm_sample = csr_array(sbm_sample)

    embed_sbm = method(n_components=2, concat=directed, svd_seed=8888)
    X_sbm = embed_sbm.fit_transform(sbm_sample)

    if directed:
        self.assertEqual(X_sbm.shape, (verts, 2 * communities))
    else:
        self.assertEqual(X_sbm.shape, (verts, communities))
    labels_pred = GaussianMixture(n_components=2).fit_predict(X_sbm)
    ari = adjusted_rand_score(labels_sbm, labels_pred)

    self.assertTrue(ari > 0.9)


class TestAdjacencySpectralEmbed(unittest.TestCase):
    def setUp(self):
        n = [10, 10]
        p = np.array([[0.9, 0.1], [0.1, 0.9]])
        wt = [[normal, poisson], [poisson, normal]]
        wtargs = [
            [dict(loc=3, scale=1), dict(lam=5)],
            [dict(lam=5), dict(loc=3, scale=1)],
        ]
        self.testgraphs = dict(
            Guw=sbm(n=n, p=p),
            Gw=sbm(n=n, p=p, wt=wt, wtargs=wtargs),
            Guwd=sbm(n=n, p=p, directed=True),
            Gwd=sbm(n=n, p=p, wt=wt, wtargs=wtargs, directed=True),
        )
        self.ase = AdjacencySpectralEmbed(n_components=2, svd_seed=9001)

    def test_output_dim(self):
        _test_output_dim(self, AdjacencySpectralEmbed)

    def test_output_dim_directed(self):
        _test_output_dim_directed(self, AdjacencySpectralEmbed)

    def test_sbm_er_binary_undirected(self):
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        _test_sbm_er_binary(self, AdjacencySpectralEmbed, P, directed=False)

    def test_sbm_er_binary_directed(self):
        P = np.array([[0.9, 0.1], [0.1, 0.9]])
        _test_sbm_er_binary(self, AdjacencySpectralEmbed, P, directed=True)

    def test_unconnected_warning(self):
        A = er_nm(100, 10)
        with pytest.warns(UserWarning):
            ase = AdjacencySpectralEmbed()
            ase.fit(A)

    def test_input_checks(self):
        with self.assertRaises(TypeError):
            ase = AdjacencySpectralEmbed(diag_aug="over 9000")
            ase.fit()

    def test_transform_closeto_fit_transform(self):
        atol = 0.2
        for diag_aug in [True, False]:
            for g, A in self.testgraphs.items():
                ase = AdjacencySpectralEmbed(
                    n_components=2, diag_aug=diag_aug, svd_seed=9001
                )
                ase.fit(A)
                Y = ase.fit_transform(A)
                if isinstance(Y, np.ndarray):
                    X = ase.transform(A)
                    np.testing.assert_allclose(X, Y, atol=atol)
                elif isinstance(Y, tuple):
                    with self.assertRaises(TypeError):
                        X = ase.transform(A)
                    X = ase.transform((A.T, A))
                    np.testing.assert_allclose(X[0], Y[0], atol=atol)
                    np.testing.assert_allclose(X[1], Y[1], atol=atol)
                else:
                    raise TypeError

    def test_transform_networkx(self):
        G = nx.grid_2d_graph(5, 5)
        ase = AdjacencySpectralEmbed(n_components=2)
        ase.fit(G)
        ase.transform(G)

    def test_transform_correct_types(self):
        ase = AdjacencySpectralEmbed(n_components=2)
        for graph in self.testgraphs.values():
            A, a = remove_vertices(graph, 1, return_removed=True)
            ase.fit(A)
            directed = ase.latent_right_ is not None
            weighted = not np.array_equal(A, A.astype(bool))
            w = ase.transform(a)
            if directed:
                self.assertIsInstance(w, tuple)
                self.assertIsInstance(w[0], np.ndarray)
                self.assertIsInstance(w[1], np.ndarray)
            elif not directed:
                self.assertIsInstance(w, np.ndarray)
                self.assertEqual(np.atleast_2d(w).shape[1], 2)

    def test_directed_vertex_direction(self):
        M = self.testgraphs["Guwd"]
        oos_idx = 0
        A, a = remove_vertices(M, indices=oos_idx, return_removed=True)
        assert_equal(np.delete(M[:, 0], oos_idx), a[0])

    def test_directed_correct_latent_positions(self):
        # setup
        ase = AdjacencySpectralEmbed(n_components=3)
        np.random.seed(8888)
        P = np.array([[0.9, 0.1, 0.1], [0.3, 0.6, 0.1], [0.1, 0.5, 0.6]])
        M, labels = sbm([200, 200, 200], P, directed=True, return_labels=True)

        # one node from each community
        oos_idx = np.nonzero(np.r_[1, np.diff(labels)[:-1]])[0]
        labels = list(labels)
        oos_labels = [labels.pop(i) for i in oos_idx]

        # Grab out-of-sample, fit, transform
        A, a = remove_vertices(M, indices=oos_idx, return_removed=True)
        latent_left, latent_right = ase.fit_transform(A)
        oos_left, oos_right = ase.transform(a)

        # separate into communities
        for i, latent in enumerate([latent_left, latent_right]):
            left = i == 0
            df = pd.DataFrame({
                "Type": labels,
                "Dimension 1": latent[:, 0],
                "Dimension 2": latent[:, 1],
                "Dimension 3": latent[:, 2],
            })
            # make sure that oos vertices are closer to their true community averages than other community averages
            means = df.groupby("Type").mean()
            if left:
                avg_dist_within = np.diag(pairwise_distances(means, oos_left))
                avg_dist_between = np.diag(pairwise_distances(means, oos_right))
                self.assertTrue(all(avg_dist_within < avg_dist_between))
            elif not left:
                avg_dist_within = np.diag(pairwise_distances(means, oos_right))
                avg_dist_between = np.diag(pairwise_distances(means, oos_left))
                self.assertTrue(all(avg_dist_within < avg_dist_between))

    def test_exceptions(self):
        ase = clone(self.ase)

        with self.assertRaises(Exception):
            ase.fit(self.testgraphs["Gwd"])
            ase.transform("9001")

        with self.assertRaises(Exception):
            Guwd = self.testgraphs["Guwd"]
            ase.fit(Guwd)
            ase.transform(np.ones(len(Guwd)))

        with self.assertRaises(ValueError):
            A, a = remove_vertices(self.testgraphs["Gw"], [0, 1], return_removed=True)
            a = a.T
            ase.fit(A)
            ase.transform(a)


class TestAdjacencySpectralEmbedSparse(unittest.TestCase):
    def test_output_dim(self):
        _test_output_dim(self, AdjacencySpectralEmbed, sparse=True)

    def test_sbm_er_binary_undirected(self):
        P = np.array([[0.8, 0.2], [0.2, 0.8]])
        _test_sbm_er_binary(
            self, AdjacencySpectralEmbed, P, directed=False, sparse=True
        )

    def test_sbm_er_binary_directed(self):
        P = np.array([[0.8, 0.2], [0.2, 0.8]])
        _test_sbm_er_binary(self, AdjacencySpectralEmbed, P, directed=True, sparse=True)

    def test_unconnected_warning(self):
        A = csr_array(er_nm(100, 10))
        with pytest.warns(UserWarning):
            ase = AdjacencySpectralEmbed()
            ase.fit(A)


class TestLaplacianSpectralEmbed(unittest.TestCase):
    def setUp(self):
        n = [10, 10]
        p = np.array([[0.9, 0.1], [0.1, 0.9]])
        wt = [[normal, poisson], [poisson, normal]]
        wtargs = [
            [dict(loc=3, scale=1), dict(lam=5)],
            [dict(lam=5), dict(loc=3, scale=1)],
        ]
        self.testgraphs = dict(
            Guw=sbm(n=n, p=p),
            Gw=sbm(n=n, p=p, wt=wt, wtargs=wtargs),
            Guwd=sbm(n=n, p=p, directed=True),
            Gwd=sbm(n=n, p=p, wt=wt, wtargs=wtargs, directed=True),
        )
        self.lse = LaplacianSpectralEmbed(n_components=2, svd_seed=9001)

    def test_different_forms(self):
        f = np.array([[1, 2], [2, 1]])
        lse = LaplacianSpectralEmbed(form="I-DAD")

    def test_transform_correct_types(self):
        lse = LaplacianSpectralEmbed(n_components=2)
        for graph in self.testgraphs.values():
            A, a = remove_vertices(graph, 1, return_removed=True)
            lse.fit(A)
            directed = lse.latent_right_ is not None
            w = lse.transform(a)
            if directed:
                self.assertIsInstance(w, tuple)
                self.assertIsInstance(w[0], np.ndarray)
                self.assertIsInstance(w[1], np.ndarray)
            elif not directed:
                self.assertIsInstance(w, np.ndarray)
                self.assertEqual(np.atleast_2d(w).shape[1], 2)

    def test_output_dim(self):
        _test_output_dim(self, LaplacianSpectralEmbed)

    def test_output_dim_directed(self):
        _test_output_dim_directed(self, LaplacianSpectralEmbed)

    def test_sbm_er_binary_undirected(self):
        P = np.array([[0.8, 0.2], [0.2, 0.3]])
        _test_sbm_er_binary(self, LaplacianSpectralEmbed, P, directed=False)

    def test_sbm_er_binary_directed(self):
        P = np.array([[0.8, 0.2], [0.2, 0.3]])
        _test_sbm_er_binary(self, LaplacianSpectralEmbed, P, directed=True)

    def test_unconnected_warning(self):
        n = [50, 50]
        p = [[1, 0], [0, 1]]
        A = sbm(n, p)
        with pytest.warns(UserWarning):
            lse = LaplacianSpectralEmbed()
            lse.fit(A)

    def test_embedding(self):
        epsilon = 0.1
        nodes_per_community = 100
        P = np.array([[0.8, 0.2], [0.2, 0.8]])
        undirected, labels_ = sbm(2 * [nodes_per_community], P, return_labels=True)

        oos_idx = 0
        A, a = remove_vertices(undirected, indices=oos_idx, return_removed=True)

        lse = LaplacianSpectralEmbed(n_components=2)
        X_hat = lse.fit_transform(A)
        w = lse.transform(a)
        self.assertTrue(X_hat[0][0] - epsilon < w[0][0] < X_hat[0][0] + epsilon)
        self.assertTrue(X_hat[0][1] - epsilon < w[0][1] < X_hat[0][1] + epsilon)


class TestLaplacianSpectralEmbedSparse(unittest.TestCase):
    def test_output_dim(self):
        _test_output_dim(self, LaplacianSpectralEmbed, sparse=True)

    def test_sbm_er_binary_undirected(self):
        P = np.array([[0.8, 0.2], [0.2, 0.3]])
        _test_sbm_er_binary(
            self, LaplacianSpectralEmbed, P, directed=False, sparse=True
        )

    def test_sbm_er_binary_directed(self):
        P = np.array([[0.8, 0.2], [0.2, 0.3]])
        _test_sbm_er_binary(self, LaplacianSpectralEmbed, P, directed=True, sparse=True)

    def test_different_forms(self):
        f = csr_array(np.array([[1, 2], [2, 1]]))
        lse = LaplacianSpectralEmbed(form="I-DAD")

    def test_unconnected_warning(self):
        n = [50, 50]
        p = [[1, 0], [0, 1]]
        A = csr_array(sbm(n, p))
        with pytest.warns(UserWarning):
            lse = LaplacianSpectralEmbed()
            lse.fit(A)


if __name__ == "__main__":
    unittest.main()
