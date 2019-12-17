import graspy.subgraph as sg
import unittest
import numpy as np
from numpy.testing import assert_equal


class TestEstimateSubgraph(unittest.TestCase):
    def test_estimate_subgraph_coh(self):
        ys = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        blank = np.ones((10, 10))
        blank[1:6, 0] = 0
        A = np.ones((10, 10, 10))

        for ind in range(10):
            if ys[ind] == 1:
                A[:, :, ind] = blank
        test_model = sg.SignalSubgraph(A, ys)
        estsub = test_model.fit_transform([5, 1])
        ver = np.ones((10, 10))
        ver[estsub] = 0
        np.testing.assert_array_equal(blank, ver)

    def test_estimate_subgraph_inc(self):
        ys = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        blank = np.ones((10, 10))
        blank[1:6, 0] = 0
        A = np.ones((10, 10, 10))

        for ind in range(10):
            if ys[ind] == 1:
                A[:, :, ind] = blank
        test_model = sg.SignalSubgraph(A, ys)
        estsub = test_model.fit_transform(5)
        ver = np.ones((10, 10))
        ver[estsub] = 0
        np.testing.assert_array_equal(blank, ver)

    def test_fit_bad_constraints(self):
        A = np.ones((5, 5, 5))
        ys = np.ones(5)
        test_model = sg.SignalSubgraph(A, ys)
        with self.assertRaises(TypeError):
            test_model.fit([1])
        with self.assertRaises(TypeError):
            test_model.fit([1, 1, 1])

    def test_construct_contingency(self):
        A = np.ones((1, 1, 5))
        A[:, :, 1::2] = 0
        ys = np.array([1, 0, 1, 0, 0])
        test_model = sg.SignalSubgraph(A, ys)
        test_model._SignalSubgraph__construct_contingency()
        cmat = test_model.contmat
        ver = np.array([[[[1, 2], [2, 0]]]], dtype=float)
        np.testing.assert_array_equal(cmat, ver)

    def test_fit_bad_type(self):
        A = [[[1 for i in range(5)] for j in range(5)] for k in range(5)]
        ys = [1, 1, 1, 1, 1]
        with self.assertRaises(TypeError):
            sg.SignalSubgraph(A, np.ones(5))
        with self.assertRaises(TypeError):
            sg.SignalSubgraph(A, set(ys))

    def test_fit_bad_size(self):
        with self.assertRaises(ValueError):
            sg.SignalSubgraph(np.ones((5, 5)), np.ones(5))
        with self.assertRaises(ValueError):
            sg.SignalSubgraph(np.ones((3, 4, 2)), np.ones(2))

    def test_fit_bad_len(self):
        A = np.ones((3, 3, 3))
        with self.assertRaises(ValueError):
            sg.SignalSubgraph(A, np.ones((3, 3)))
        with self.assertRaises(ValueError):
            sg.SignalSubgraph(A, np.array([0, 1, 2]))
        with self.assertRaises(ValueError):
            sg.SignalSubgraph(A, np.ones(2))
