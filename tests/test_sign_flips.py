# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np

from graspologic.align import SignFlips


class TestSignFlips(unittest.TestCase):
    def test_bad_kwargs(self):
        with self.assertRaises(TypeError):
            SignFlips(criterion={"this is a": "dict"})
        with self.assertRaises(ValueError):
            SignFlips(criterion="cep")
        # check delayed ValueError
        with self.assertRaises(ValueError):
            aligner = SignFlips(criterion="median")
            X = np.arange(6).reshape(6, 1)
            Y = np.arange(6).reshape(6, 1)
            aligner.criterion = "something"
            aligner.fit(X, Y)

    def test_bad_datasets(self):
        X = np.arange(6).reshape(6, 1)
        Y = np.arange(6).reshape(6, 1)
        Y_wrong_d = np.arange(12).reshape(6, 2)
        # check passing weird stuff as input (caught by us)
        with self.assertRaises(TypeError):
            aligner = SignFlips()
            aligner.fit("hello there", Y)
        with self.assertRaises(TypeError):
            aligner = SignFlips()
            aligner.fit(X, "hello there")
        with self.assertRaises(TypeError):
            aligner = SignFlips()
            aligner.fit({"hello": "there"}, Y)
        with self.assertRaises(TypeError):
            aligner = SignFlips()
            aligner.fit(X, {"hello": "there"})
        # check passing arrays of weird ndims (caught by check_array)
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit(X, Y.reshape(3, 2, 1))
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit(X.reshape(3, 2, 1), Y)
        # check passing arrays with different dimensions (caught by us)
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit(X, Y_wrong_d)
        # check passing array with wrong dimensions to transform (caught by us)
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit(X, Y)
            aligner.transform(Y_wrong_d)

    def test_two_datasets(self):
        X = np.arange(6).reshape(3, 2) * (-1)
        Y = np.arange(6).reshape(3, 2) @ np.diag([1, -1]) + 0.5
        # X flips sign in the first dimension
        Q_answer = np.array([[-1, 0], [0, 1]])
        X_answer = X.copy() @ Q_answer
        # first, do fit and transform separately
        aligner_1 = SignFlips()
        aligner_1.fit(X, Y)
        Q_test = aligner_1.Q_
        X_test = aligner_1.transform(X)
        self.assertTrue(np.all(Q_test == Q_answer))
        self.assertTrue(np.all(X_test == X_answer))
        # now, do fit_transform
        aligner_2 = SignFlips()
        X_test = aligner_2.fit_transform(X, Y)
        Q_test = aligner_2.Q_
        self.assertTrue(np.all(Q_test == Q_answer))
        self.assertTrue(np.all(X_test == X_answer))
        # try giving a different matrix as the sole input (I)
        I_test = aligner_2.transform(np.eye(2))
        I_answer = np.diag([-1, 1])
        self.assertTrue(np.all(I_test == I_answer))

    def test_max_criterion(self):
        X = np.arange(6).reshape(3, 2) * (-1)
        Y = np.arange(6).reshape(3, 2) @ np.diag([1, -1]) + 0.5
        # in this case, Y should be unchanged, and X matched to Y
        # so X flips sign in the first dimension
        Q_answer = np.array([[-1, 0], [0, 1]])
        X_answer = X.copy() @ Q_answer
        # set criterion to "max", see if that works
        aligner = SignFlips(criterion="max")
        aligner.fit(X, Y)
        Q_test = aligner.Q_
        X_test = aligner.transform(X)
        self.assertTrue(np.all(Q_test == Q_answer))
        self.assertTrue(np.all(X_test == X_answer))


if __name__ == "__main__":
    unittest.main()
