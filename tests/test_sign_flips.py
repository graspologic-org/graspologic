# Anton Alyakin
# aalyaki1 [at] jhu.edu
# 09.01.2020

import unittest

import numpy as np

from graspy.align import SignFlips


class TestSignFlips(unittest.TestCase):
    def test_bad_kwargs(self):
        with self.assertRaises(TypeError):
            SignFlips(freeze_Y="oops")
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
        # check passing weird stuff as input (caught by check_array)
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit_transform("hello there", Y)
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit_transform(X, "hello there")
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit_transform({"hello": "there"}, Y)
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit_transform(X, {"hello": "there"})
        # check passing arrays of weird ndims (caught by check_array)
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit_transform(X, Y.reshape(3, 2, 1))
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit_transform(X.reshape(3, 2, 1), Y)
        # check passing arrays with different (catching ourselves)
        with self.assertRaises(ValueError):
            aligner = SignFlips()
            aligner.fit_transform(X, Y.T)

    def test_freeze_Y_true_two_datasets(self):
        X = np.arange(6).reshape(3, 2) * (-1)
        Y = np.arange(6).reshape(3, 2) @ np.diag([1, -1]) + 0.5
        # in this case, Y should be unchanged, and X matched to Y
        # so X flips sign in the first dimension
        Q_X_answer = np.array([[-1, 0], [0, 1]])
        Q_Y_answer = np.eye(2)
        X_answer = X.copy() @ Q_X_answer
        Y_answer = Y.copy()
        # first, do fit and transform separately
        aligner_1 = SignFlips(freeze_Y=True)
        aligner_1.fit(X, Y)
        Q_X_test, Q_Y_test = aligner_1.Q_X, aligner_1.Q_Y
        X_test, Y_test = aligner_1.transform(X, Y)
        self.assertTrue(np.all(Q_X_test == Q_X_answer))
        self.assertTrue(np.all(Q_Y_test == Q_Y_answer))
        self.assertTrue(np.all(X_test == X_answer))
        self.assertTrue(np.all(Y_test == Y_answer))
        # now, do fit_transform
        aligner_2 = SignFlips(freeze_Y=True)
        X_test, Y_test = aligner_2.fit_transform(X, Y)
        Q_X_test, Q_Y_test = aligner_2.Q_X, aligner_2.Q_Y
        self.assertTrue(np.all(Q_X_test == Q_X_answer))
        self.assertTrue(np.all(Q_Y_test == Q_Y_answer))
        self.assertTrue(np.all(X_test == X_answer))
        self.assertTrue(np.all(Y_test == Y_answer))

    def test_enforce_Y_false_two_datasets(self):
        X = np.arange(6).reshape(3, 2) * (-1)
        Y = np.arange(6).reshape(3, 2) @ np.diag([1, -1]) + 0.5
        # in this case, all need to become positive,
        # so X flips all sings, and Y flips the second dimension
        Q_X_answer = np.eye(2) * (-1)
        Q_Y_answer = np.array([[1, 0], [0, -1]])
        X_answer = X @ Q_X_answer
        Y_answer = Y @ Q_Y_answer
        # first, do fit and transform separately
        aligner_1 = SignFlips(freeze_Y=False)
        aligner_1.fit(X, Y)
        Q_X_test, Q_Y_test = aligner_1.Q_X, aligner_1.Q_Y
        X_test, Y_test = aligner_1.transform(X, Y)
        self.assertTrue(np.all(Q_X_test == Q_X_answer))
        self.assertTrue(np.all(Q_Y_test == Q_Y_answer))
        self.assertTrue(np.all(X_test == X_answer))
        self.assertTrue(np.all(Y_test == Y_answer))
        # now, do fit_transform
        aligner_2 = SignFlips(freeze_Y=False)
        X_test, Y_test = aligner_2.fit_transform(X, Y)
        Q_X_test, Q_Y_test = aligner_2.Q_X, aligner_2.Q_Y
        self.assertTrue(np.all(Q_X_test == Q_X_answer))
        self.assertTrue(np.all(Q_Y_test == Q_Y_answer))
        self.assertTrue(np.all(X_test == X_answer))
        self.assertTrue(np.all(Y_test == Y_answer))

    def test_freeze_Y_true_one_dataset(self):
        X = np.arange(6).reshape(3, 2) * (-1)
        Y = np.arange(6).reshape(3, 2) @ np.diag([1, -1]) + 0.5

        # first, try only providing X to fit
        # (should fail, because we are trying to match X to Y)
        aligner_1 = SignFlips(freeze_Y=True)
        with self.assertRaises(ValueError):
            aligner_1.fit(X)

        # now, fit to both, but only provide one dataset to transform
        aligner_2 = SignFlips(freeze_Y=True)
        aligner_2.fit(X, Y)
        # try giving X as the sole input
        X_test = aligner_2.transform(X)
        X_answer = X @ np.diag([-1, 1])
        self.assertTrue(np.all(X_test == X_answer))
        # try giving a different matrix as the sole input (I)
        I_test = aligner_2.transform(np.eye(2))
        I_answer = np.diag([-1, 1])
        self.assertTrue(np.all(I_test == I_answer))

    def test_enforce_Y_false_one_dataset(self):
        X = np.arange(6).reshape(3, 2) * (-1)
        Y = np.arange(6).reshape(3, 2) @ np.diag([1, -1]) + 0.5

        X_answer = X.copy() * (-1)
        Y_answer = Y.copy() @ np.diag([1, -1])
        # first, try only providing X to fit
        # (should work, because it's just a flip to positive)
        aligner_1 = SignFlips(freeze_Y=False)
        X_test = aligner_1.fit_transform(X)
        self.assertTrue(np.all(X_test == X_answer))
        negative_I_test = aligner_1.transform(-np.eye(2))
        self.assertTrue(np.all(negative_I_test == negative_I_test))
        # now, provide both
        #  they both should be flipped to positive
        aligner_2 = SignFlips(freeze_Y=False)
        X_test, Y_test = aligner_2.fit_transform(X, Y)
        self.assertTrue(np.all(X_test == X_answer))
        self.assertTrue(np.all(Y_test == Y_answer))

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
        aligner = SignFlips(freeze_Y=True, criteria="max")
        aligner.fit(X, Y)
        Q_X_test, Q_Y_test = aligner.Q_X, aligner.Q_Y
        X_test, Y_test = aligner.transform(X, Y)
        self.assertTrue(np.all(Q_X_test == Q_X_answer))
        self.assertTrue(np.all(Q_Y_test == Q_Y_answer))
        self.assertTrue(np.all(X_test == X_answer))
        self.assertTrue(np.all(Y_test == Y_answer))


if __name__ == "__main__":
    unittest.main()
