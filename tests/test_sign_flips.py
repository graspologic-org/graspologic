# Anton Alyakin
# aalyaki1 [at] jhu.edu
# 09.01.2020

import unittest

import numpy as np

from graspy.align import SignFlips


class TestSignFlips(unittest.TestCase):
    def test_bad_kwargs(self):
        with self.assertRaises(TypeError):
            SignFlips(criteria={"this is a": "dict"})
        with self.assertRaises(ValueError):
            SignFlips(criteria="cep")
        # check delayed ValueError
        with self.assertRaises(ValueError):
            aligner = SignFlips(criteria="median")
            X = np.arange(6).reshape(6, 1)
            Y = np.arange(6).reshape(6, 1)
            aligner.criteria = "something"
            aligner.fit(X, Y)

    def test_bad_datasets(self):
        X = np.arange(6).reshape(6, 1)
        Y = np.arange(6).reshape(6, 1)
        Y_wrong_d = np.arange(12).reshape(6, 2)
        # check passing weird stuff as input (caught by us)
        with self.assertRaises(TypeError):
            aligner = SignFlips()
            aligner.fit_transform("hello there", Y)
        with self.assertRaises(TypeError):
            aligner = SignFlips()
            aligner.fit_transform(X, "hello there")
        with self.assertRaises(TypeError):
            aligner = SignFlips()
            aligner.fit_transform({"hello": "there"}, Y)
        with self.assertRaises(TypeError):
            aligner = SignFlips()
            aligner.fit_transform(X, {"hello": "there"})
        # check passing arrays of weird ndims (caught by check_array)
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit_transform(X, Y.reshape(3, 2, 1))
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit_transform(X.reshape(3, 2, 1), Y)
        # check passing arrays with different dimensions (caught by us)
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit_transform(X, Y_wrong_d)

    def test_two_datasets(self):
        X = np.arange(6).reshape(3, 2) * (-1)
        Y = np.arange(6).reshape(3, 2) @ np.diag([1, -1]) + 0.5
        # in this case, Y should be unchanged, and X matched to Y
        # so X flips sign in the first dimension
        Q_X_answer = np.array([[-1, 0], [0, 1]])
        Q_Y_answer = np.eye(2)
        X_answer = X.copy() @ Q_X_answer
        Y_answer = Y.copy()
        # first, do fit and transform separately
        aligner_1 = SignFlips()
        aligner_1.fit(X, Y)
        Q_X_test, Q_Y_test = aligner_1.Q_X, aligner_1.Q_Y
        X_test, Y_test = aligner_1.transform(X, Y)
        self.assertTrue(np.all(Q_X_test == Q_X_answer))
        self.assertTrue(np.all(Q_Y_test == Q_Y_answer))
        self.assertTrue(np.all(X_test == X_answer))
        self.assertTrue(np.all(Y_test == Y_answer))
        # now, do fit_transform
        aligner_2 = SignFlips()
        X_test, Y_test = aligner_2.fit_transform(X, Y)
        Q_X_test, Q_Y_test = aligner_2.Q_X, aligner_2.Q_Y
        self.assertTrue(np.all(Q_X_test == Q_X_answer))
        self.assertTrue(np.all(Q_Y_test == Q_Y_answer))
        self.assertTrue(np.all(X_test == X_answer))
        self.assertTrue(np.all(Y_test == Y_answer))

    def test_one_dataset(self):
        X = np.arange(6).reshape(3, 2) * (-1)
        Y = np.arange(6).reshape(3, 2) @ np.diag([1, -1]) + 0.5
        # fit to both, but only provide one dataset to transform
        aligner = SignFlips()
        aligner.fit(X, Y)
        # try giving X as the sole input
        X_test = aligner.transform(X)
        X_answer = X @ np.diag([-1, 1])
        self.assertTrue(np.all(X_test == X_answer))
        # try giving a different matrix as the sole input (I)
        I_test = aligner.transform(np.eye(2))
        I_answer = np.diag([-1, 1])
        self.assertTrue(np.all(I_test == I_answer))

    def test_max_criteria(self):
        X = np.arange(6).reshape(3, 2) * (-1)
        Y = np.arange(6).reshape(3, 2) @ np.diag([1, -1]) + 0.5
        # in this case, Y should be unchanged, and X matched to Y
        # so X flips sign in the first dimension
        Q_X_answer = np.array([[-1, 0], [0, 1]])
        Q_Y_answer = np.eye(2)
        X_answer = X.copy() @ Q_X_answer
        Y_answer = Y.copy()
        # set criteria to "max", see if that works
        aligner = SignFlips(criteria="max")
        aligner.fit(X, Y)
        Q_X_test, Q_Y_test = aligner.Q_X, aligner.Q_Y
        X_test, Y_test = aligner.transform(X, Y)
        self.assertTrue(np.all(Q_X_test == Q_X_answer))
        self.assertTrue(np.all(Q_Y_test == Q_Y_answer))
        self.assertTrue(np.all(X_test == X_answer))
        self.assertTrue(np.all(Y_test == Y_answer))


if __name__ == "__main__":
    unittest.main()
