# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import pytest
import unittest
import numpy as np
import networkx as nx
from sklearn.metrics import pairwise_distances

from graspologic.embed import AdjacencySpectralEmbed
from graspologic.inference import LatentDistributionTest
from graspologic.simulations import er_np, sbm


class TestLatentDistributionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        np.random.seed(123456)
        cls.tests = {"dcorr": "euclidean", "hsic": "gaussian", "mgc": "euclidean"}
        cls.A1 = er_np(20, 0.3)
        cls.A2 = er_np(20, 0.3)

    def test_fit_ase_works(self):
        for test in self.tests.keys():
            ldt = LatentDistributionTest(test, self.tests[test], n_bootstraps=10)
            assert ldt.fit(self.A1, self.A2) is ldt

    def test_fit_predict_ase_works(self):
        for test in self.tests:
            ldt = LatentDistributionTest(test, self.tests[test], n_bootstraps=10)
            p = ldt.fit_predict(self.A1, self.A2)
            assert float(p) <= 1 and float(p) >= 0

    def test_workers(self):
        ldt = LatentDistributionTest("dcorr", "euclidean", n_bootstraps=4, workers=4)
        ldt.fit(self.A1, self.A2)

    def test_callable_metric(self):
        def metric_func(X, Y=None, workers=None):
            return pairwise_distances(X, metric="euclidean") * 0.5

        ldt = LatentDistributionTest("dcorr", metric_func, n_bootstraps=10)
        ldt.fit(self.A1, self.A2)

    def test_bad_kwargs(self):
        # check test argument
        with self.assertRaises(TypeError):
            LatentDistributionTest(test=0)
        with self.assertRaises(ValueError):
            LatentDistributionTest(test="foo")
        # check metric argument
        with self.assertRaises(TypeError):
            LatentDistributionTest(metric=0)
        with self.assertRaises(ValueError):
            LatentDistributionTest(metric="some_kind_of_kernel")
        # check metric argument modified
        with self.assertRaises(TypeError):
            ldt = LatentDistributionTest()
            ldt.metric = 0
            ldt.fit(self.A1, self.A2)
        with self.assertRaises(ValueError):
            ldt = LatentDistributionTest()
            ldt.metric = "some_kind_of_kernel"
            ldt.fit(self.A1, self.A2)
        # check n_components argument
        with self.assertRaises(TypeError):
            LatentDistributionTest(n_components=0.5)
        with self.assertRaises(ValueError):
            LatentDistributionTest(n_components=-100)
        # check n_bootstraps argument
        with self.assertRaises(TypeError):
            LatentDistributionTest(n_bootstraps=0.5)
        with self.assertRaises(ValueError):
            LatentDistributionTest(n_bootstraps=-100)
        # check workers argument
        with self.assertRaises(TypeError):
            LatentDistributionTest(workers=0.5)
        # check size_correction argument
        with self.assertRaises(TypeError):
            LatentDistributionTest(size_correction=0)
        # check pooled argument
        with self.assertRaises(TypeError):
            LatentDistributionTest(pooled=0)
        # check align_type argument
        with self.assertRaises(ValueError):
            LatentDistributionTest(align_type="foo")
        with self.assertRaises(TypeError):
            LatentDistributionTest(align_type={"not a": "string"})
        # check align_kws argument
        with self.assertRaises(TypeError):
            LatentDistributionTest(align_kws="foo")
        # check input_graph argument
        with self.assertRaises(TypeError):
            LatentDistributionTest(input_graph="hello")

    def test_n_bootstraps(self):
        for test in self.tests.keys():
            ldt = LatentDistributionTest(test, self.tests[test], n_bootstraps=123)
            ldt.fit(self.A1, self.A2)
            self.assertEqual(ldt.null_distribution_.shape[0], 123)

    def test_passing_networkx(self):
        np.random.seed(123)
        A1 = er_np(20, 0.8)
        A2 = er_np(20, 0.8)
        A1_nx = nx.from_numpy_matrix(A1)
        A2_nx = nx.from_numpy_matrix(A2)
        # check passing nx, when exepect embeddings
        with self.assertRaises(TypeError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict(A1_nx, A2)
        with self.assertRaises(TypeError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict(A1, A2_nx)
        with self.assertRaises(TypeError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict(A1_nx, A2_nx)
        # check that the appropriate input works
        ldt = LatentDistributionTest(input_graph=True)
        ldt.fit_predict(A1_nx, A2_nx)

    def test_passing_embeddings(self):
        np.random.seed(123)
        A1 = er_np(20, 0.8)
        A2 = er_np(20, 0.8)
        ase_1 = AdjacencySpectralEmbed(n_components=2)
        X1 = ase_1.fit_transform(A1)
        ase_2 = AdjacencySpectralEmbed(n_components=2)
        X2 = ase_2.fit_transform(A2)
        ase_3 = AdjacencySpectralEmbed(n_components=1)
        X3 = ase_3.fit_transform(A2)
        # check embeddings having weird ndim
        with self.assertRaises(ValueError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict(X1, X2.reshape(-1, 1, 1))
        with self.assertRaises(ValueError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict(X1.reshape(-1, 1, 1), X2)
        # check embeddings having mismatching number of components
        with self.assertRaises(ValueError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict(X1, X3)
        with self.assertRaises(ValueError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict(X3, X1)
        # check passing weird stuff as input (caught by us)
        with self.assertRaises(TypeError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict("hello there", X1)
        with self.assertRaises(TypeError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict(X1, "hello there")
        with self.assertRaises(TypeError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict({"hello": "there"}, X1)
        with self.assertRaises(TypeError):
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict(X1, {"hello": "there"})
        # check passing infinite in input (caught by check_array)
        with self.assertRaises(ValueError):
            X1_w_inf = X1.copy()
            X1_w_inf[1, 1] = np.inf
            ldt = LatentDistributionTest(input_graph=False)
            ldt.fit_predict(X1_w_inf, X2)
        # check that the appropriate input works
        ldt = LatentDistributionTest(input_graph=False)
        ldt.fit_predict(X1, X2)

    def test_pooled(self):
        np.random.seed(123)
        A1 = er_np(20, 0.3)
        A2 = er_np(100, 0.3)

        ldt = LatentDistributionTest(pooled=True)
        ldt.fit(A1, A2)

    def test_distances_and_kernels(self):
        np.random.seed(123)
        A1 = er_np(20, 0.3)
        A2 = er_np(100, 0.3)
        # some valid combinations of test and metric
        # # would love to do this, but currently FutureWarning breaks this
        # with pytest.warns(None) as record:
        #     for test in self.tests.keys():
        #         ldt = LatentDistributionTest(test, self.tests[test])
        #         ldt.fit(A1, A2)
        #     ldt = LatentDistributionTest("hsic", "rbf")
        #     ldt.fit(A1, A2)
        # assert len(record) == 0
        for test in self.tests.keys():
            ldt = LatentDistributionTest(test, self.tests[test])
            ldt.fit(A1, A2)
        ldt = LatentDistributionTest("hsic", "rbf")
        ldt.fit(A1, A2)
        # some invalid combinations of test and metric
        with pytest.warns(UserWarning):
            ldt = LatentDistributionTest("hsic", "euclidean")
        with pytest.warns(UserWarning):
            ldt = LatentDistributionTest("dcorr", "gaussian")
        with pytest.warns(UserWarning):
            ldt = LatentDistributionTest("dcorr", "rbf")

    def test_bad_matrix_inputs(self):
        ldt = LatentDistributionTest("dcorr")

        bad_matrix = [[1, 2]]
        with self.assertRaises(TypeError):
            ldt.fit(bad_matrix, self.A2)

    def test_directed_inputs(self):
        np.random.seed(2)
        A = er_np(100, 0.3, directed=True)
        B = er_np(100, 0.3, directed=True)
        C = er_np(100, 0.3, directed=False)

        # two directed graphs is okay
        ldt = LatentDistributionTest()
        ldt.fit(A, B)

        # an undirected and a direced graph is not okay
        with self.assertRaises(ValueError):
            ldt.fit(A, C)
        with self.assertRaises(ValueError):
            ldt.fit(C, B)

    def test_SBM_dcorr(self):
        for test in self.tests.keys():
            np.random.seed(12345678)
            B1 = np.array([[0.5, 0.2], [0.2, 0.5]])

            B2 = np.array([[0.7, 0.2], [0.2, 0.7]])
            b_size = 200
            A1 = sbm(2 * [b_size], B1)
            A2 = sbm(2 * [b_size], B1)
            A3 = sbm(2 * [b_size], B2)

            ldt_null = LatentDistributionTest(
                test, self.tests[test], n_components=2, n_bootstraps=100
            )
            ldt_alt = LatentDistributionTest(
                test, self.tests[test], n_components=2, n_bootstraps=100
            )
            p_null = ldt_null.fit_predict(A1, A2)
            p_alt = ldt_alt.fit_predict(A1, A3)
            self.assertTrue(p_null > 0.05)
            self.assertTrue(p_alt <= 0.05)

    def test_different_sizes_null(self):
        np.random.seed(314)

        A1 = er_np(100, 0.8)
        A2 = er_np(1000, 0.8)

        ldt_not_corrected = LatentDistributionTest(
            "hsic", "gaussian", n_components=2, n_bootstraps=100, size_correction=False
        )
        ldt_corrected_1 = LatentDistributionTest(
            "hsic", "gaussian", n_components=2, n_bootstraps=100, size_correction=True
        )
        ldt_corrected_2 = LatentDistributionTest(
            "hsic", "gaussian", n_components=2, n_bootstraps=100, size_correction=True
        )

        p_not_corrected = ldt_not_corrected.fit_predict(A1, A2)
        p_corrected_1 = ldt_corrected_1.fit_predict(A1, A2)
        p_corrected_2 = ldt_corrected_2.fit_predict(A2, A1)

        self.assertTrue(p_not_corrected <= 0.05)
        self.assertTrue(p_corrected_1 > 0.05)
        self.assertTrue(p_corrected_2 > 0.05)

    def test_different_sizes_null(self):
        np.random.seed(314)

        A1 = er_np(100, 0.8)
        A2 = er_np(1000, 0.7)

        ldt_corrected_1 = LatentDistributionTest(
            "hsic", "gaussian", n_components=2, n_bootstraps=100, size_correction=True
        )
        ldt_corrected_2 = LatentDistributionTest(
            "hsic", "gaussian", n_components=2, n_bootstraps=100, size_correction=True
        )

        p_corrected_1 = ldt_corrected_1.fit_predict(A1, A2)
        p_corrected_2 = ldt_corrected_2.fit_predict(A2, A1)

        self.assertTrue(p_corrected_1 <= 0.05)
        self.assertTrue(p_corrected_2 <= 0.05)

    def test_different_aligners(self):
        np.random.seed(314)
        A1 = er_np(100, 0.8)
        A2 = er_np(100, 0.8)
        ase_1 = AdjacencySpectralEmbed(n_components=2)
        X1 = ase_1.fit_transform(A1)
        ase_2 = AdjacencySpectralEmbed(n_components=2)
        X2 = ase_2.fit_transform(A2)
        X2 = -X2

        ldt_1 = LatentDistributionTest(input_graph=False, align_type=None)
        p_val_1 = ldt_1.fit_predict(X1, X2)
        self.assertTrue(p_val_1 < 0.05)

        ldt_2 = LatentDistributionTest(input_graph=False, align_type="sign_flips")
        p_val_2 = ldt_2.fit_predict(X1, X2)
        self.assertTrue(p_val_2 >= 0.05)

        # also checking that kws are passed through
        ldt_3 = LatentDistributionTest(
            input_graph=False,
            align_type="seedless_procrustes",
            align_kws={"init": "sign_flips"},
        )
        p_val_3 = ldt_3.fit_predict(X1, X2)
        self.assertTrue(p_val_3 >= 0.05)


if __name__ == "__main__":
    unittest.main()
