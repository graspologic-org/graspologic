# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np

from graspologic.align import OrthogonalProcrustes


class TestOrthogonalProcrustes(unittest.TestCase):
    def test_bad_datasets(self):
        X = np.arange(6).reshape(6, 1)
        Y = np.arange(6).reshape(6, 1)
        Y_wrong_d = np.arange(12).reshape(6, 2)
        Y_wrong_n = np.arange(12).reshape(12, 1)
        # check passing weird stuff as input (caught by us)
        with self.assertRaises(TypeError):
            aligner = OrthogonalProcrustes()
            aligner.fit("hello there", Y)
        with self.assertRaises(TypeError):
            aligner = OrthogonalProcrustes()
            aligner.fit(X, "hello there")
        with self.assertRaises(TypeError):
            aligner = OrthogonalProcrustes()
            aligner.fit({"hello": "there"}, Y)
        with self.assertRaises(TypeError):
            aligner = OrthogonalProcrustes()
            aligner.fit(X, {"hello": "there"})
        # check passing arrays of weird ndims (caught by check_array)
        with self.assertRaises(ValueError):
            aligner = OrthogonalProcrustes()
            aligner.fit(X, Y.reshape(3, 2, 1))
        with self.assertRaises(ValueError):
            aligner = OrthogonalProcrustes()
            aligner.fit(X.reshape(3, 2, 1), Y)
        # check passing arrays with different dimensions (caught by us)
        with self.assertRaises(ValueError):
            aligner = OrthogonalProcrustes()
            aligner.fit(X, Y_wrong_d)
        # check passing arrays with different number of vertices (caught by us)
        with self.assertRaises(ValueError):
            aligner = OrthogonalProcrustes()
            aligner.fit(X, Y_wrong_n)
        # check passing array with wrong dimensions to transform (caught by us)
        with self.assertRaises(ValueError):
            aligner = OrthogonalProcrustes()
            aligner.fit(X, Y)
            aligner.transform(Y_wrong_d)
        # passing array with different number of vertices to fit is okay
        aligner = OrthogonalProcrustes()
        aligner.fit(X, Y)
        aligner.transform(Y_wrong_n)

    def test_identity(self):
        Y = np.array([[1234, 19], [6798, 18], [9876, 17], [4321, 16]])

        aligner = OrthogonalProcrustes()
        aligner.fit(Y, Y)

        assert np.all(np.isclose(aligner.Q_, np.eye(2)))

    def test_two_datasets(self):
        # A very simple example with a true existing solution
        #       X:                 Y:
        #            |                  |
        #            |                  2
        #            |   1              |
        #        2   |                  |
        #            |                  |
        #            |                  |
        #      ------+------      -1----+----3-
        #            |                  |
        #            |                  |
        #            |    4             |
        #         3  |                  |
        #            |                  4
        #            |                  |
        #
        # solution is
        #  _                 _             _      _
        # |                   |           |        |
        # | 3 / 5     - 4 / 5 |           | -1   0 |
        # |                   |   times   |        |
        # | 3 / 5       3 / 5 |           |  0   1 |
        # |_                 _|           |_      _|
        # because it is just rotation times reflection
        X = np.array([[3, 4], [-4, 3], [-3, -4], [4, -3]])
        Y = np.array([[-5, 0], [0, 5], [5, 0], [0, -5]])
        Q_answer = np.array([[-0.6, -0.8], [-0.8, 0.6]])
        X_answer = X.copy() @ Q_answer

        # first, do fit and transform separately
        aligner_1 = OrthogonalProcrustes()
        aligner_1.fit(X, Y)
        Q_test_1 = aligner_1.Q_
        X_test_1 = aligner_1.transform(X)
        self.assertTrue(np.isclose(0, aligner_1.score_))
        self.assertTrue(np.all(np.isclose(Q_test_1, Q_answer)))
        self.assertTrue(np.all(np.isclose(X_test_1, X_answer)))
        # now, do fit_transform
        aligner_2 = OrthogonalProcrustes()
        X_test_2 = aligner_2.fit_transform(X, Y)
        Q_test_2 = aligner_2.Q_
        self.assertTrue(np.isclose(0, aligner_2.score_))
        self.assertTrue(np.all(np.isclose(Q_test_2, Q_answer)))
        self.assertTrue(np.all(np.isclose(X_test_2, X_answer)))


if __name__ == "__main__":
    unittest.main()
