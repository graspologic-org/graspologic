# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest
import numpy as np
from graspy.utils import pass_to_ranks, is_unweighted


class TestPTR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loopless_undirected_input = np.array(
            [[0, 1, 60, 60], [1, 0, 400, 0], [60, 400, 0, 80], [60, 0, 80, 0]]
        )
        cls.looped_undirected_input = np.array(
            [[0.5, 1, 60, 60], [1, 0, 400, 0], [60, 400, 20, 80], [60, 0, 80, 0]]
        )
        cls.loopless_directed_input = np.array(
            [[0, 1, 60, 60], [1, 0, 400, 0], [3, 600, 0, 80], [20, 0, 401, 0]]
        )
        cls.looped_directed_input = np.array(
            [[21, 1, 60, 60], [1, 0, 400, 0], [3, 600, 0, 80], [20, 0, 401, 30]]
        )

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            pass_to_ranks(self.loopless_undirected_input, method="hazelnutcoffe")
        with self.assertRaises(TypeError):
            pass_to_ranks("hi", "hi")

    def test_zeroboost_loopless_undirected(self):
        ptr_expected = np.array(
            [
                [0, 2.0 / 6, 3.5 / 6, 3.5 / 6],
                [2.0 / 6, 0, 1, 0],
                [3.5 / 6, 1, 0, 5.0 / 6],
                [3.5 / 6, 0, 5.0 / 6, 0],
            ]
        )

        _run_test(self, "zero-boost", self.loopless_undirected_input, ptr_expected)

    def test_zeroboost_looped_undirected(self):
        ptr_expected = np.array(
            [
                [0.4, 0.5, 0.75, 0.75],
                [0.5, 0, 1, 0],
                [0.75, 1, 0.6, 0.9],
                [0.75, 0, 0.9, 0],
            ]
        )

        _run_test(self, "zero-boost", self.looped_undirected_input, ptr_expected)

    def test_zeroboost_loopless_directed(self):
        ptr_expected = np.array(
            [[0, 3.5, 7.5, 7.5], [3.5, 0, 10, 0], [5, 12, 0, 9], [6, 0, 11, 0]]
        )
        ptr_expected /= 12
        _run_test(self, "zero-boost", self.loopless_directed_input, ptr_expected)

    def test_zeroboost_looped_directed(self):
        ptr_expected = np.array(
            [[9, 5.5, 11.5, 11.5], [5.5, 0, 14, 0], [7, 16, 0, 13], [8, 0, 15, 10]]
        )
        ptr_expected /= 16
        _run_test(self, "zero-boost", self.looped_directed_input, ptr_expected)

    def test_simpleall_loopless_undirected(self):
        ptr_expected = 0.5 * np.array(
            [
                [0, 0.1764706, 0.5294118, 0.5294118],
                [0.1764706, 0, 1.1176471, 0],
                [0.5294118, 1.1176471, 0, 0.8823529],
                [0.5294118, 0, 0.8823529, 0],
            ]
        )

        _run_test(self, "simple-all", self.loopless_undirected_input, ptr_expected)

    def test_simpleall_looped_undirected(self):
        ptr_expected = 0.5 * np.array(
            [
                [0.1176471, 0.2941176, 0.7647059, 0.7647059],
                [0.2941176, 0, 1.3529412, 0],
                [0.7647059, 1.3529412, 0.4705882, 1.1176471],
                [0.7647059, 0, 1.1176471, 0],
            ]
        )

        _run_test(self, "simple-all", self.looped_undirected_input, ptr_expected)

    def test_simpleall_loopless_directed(self):
        ptr_expected = 0.5 * np.array(
            [
                [0, 0.1764706, 0.6470588, 0.6470588],
                [0.1764706, 0, 0.9411765, 0],
                [0.3529412, 1.1764706, 0, 0.8235294],
                [0.4705882, 0, 1.0588235, 0],
            ]
        )
        _run_test(self, "simple-all", self.loopless_directed_input, ptr_expected)

    def test_simpleall_looped_directed(self):
        ptr_expected = 0.5 * np.array(
            [
                [0.5882353, 0.1764706, 0.8823529, 0.8823529],
                [0.1764706, 0, 1.1764706, 0],
                [0.3529412, 1.4117647, 0, 1.0588235],
                [0.4705882, 0, 1.2941176, 0.7058824],
            ]
        )
        _run_test(self, "simple-all", self.looped_directed_input, ptr_expected)

    def test_simplenonzero_loopless_undirected(self):
        ptr_expected = 0.5 * np.array(
            [
                [0, 0.2727273, 0.8181818, 0.8181818],
                [0.2727273, 0, 1.7272727, 0],
                [0.8181818, 1.7272727, 0, 1.3636364],
                [0.8181818, 0, 1.3636364, 0],
            ]
        )

        _run_test(self, "simple-nonzero", self.loopless_undirected_input, ptr_expected)

    def test_simplenonzero_looped_undirected(self):
        ptr_expected = 0.5 * np.array(
            [
                [0.1538462, 0.3846154, 1, 1],
                [0.3846154, 0, 1.7692308, 0],
                [1, 1.7692308, 0.6153846, 1.4615385],
                [1, 0, 1.4615385, 0],
            ]
        )

        _run_test(self, "simple-nonzero", self.looped_undirected_input, ptr_expected)

    def test_simplenonzero_loopless_directed(self):
        ptr_expected = 0.5 * np.array(
            [
                [0, 0.2727273, 1, 1],
                [0.2727273, 0, 1.4545455, 0],
                [0.5454545, 1.8181818, 0, 1.2727273],
                [0.7272727, 0, 1.6363636, 0],
            ]
        )

        _run_test(self, "simple-nonzero", self.loopless_directed_input, ptr_expected)

    def test_simplenonzero_looped_directed(self):
        ptr_expected = 0.5 * np.array(
            [
                [0.7692308, 0.2307692, 1.1538462, 1.1538462],
                [0.2307692, 0, 1.5384615, 0],
                [0.4615385, 1.8461538, 0, 1.3846154],
                [0.6153846, 0, 1.6923077, 0.9230769],
            ]
        )

        _run_test(self, "simple-nonzero", self.looped_directed_input, ptr_expected)


def _run_test(self, methodstr, input_graph, expected):
    ptr_out = pass_to_ranks(input_graph, method=methodstr)
    self.assertTrue(np.allclose(ptr_out, expected, rtol=1e-04))


if __name__ == "__main__":
    unittest.main()
