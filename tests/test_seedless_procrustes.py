# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np
from scipy import stats
from graspologic.align import SignFlips
from graspologic.align import SeedlessProcrustes


class TestSeedlessProcrustes(unittest.TestCase):
    def test_bad_kwargs(self):
        # type errors for all but initial Q and initial P
        with self.assertRaises(TypeError):
            SeedlessProcrustes(optimal_transport_lambda="oops")
        with self.assertRaises(TypeError):
            SeedlessProcrustes(optimal_transport_eps="oops")
        with self.assertRaises(TypeError):
            SeedlessProcrustes(optimal_transport_num_reps=3.14)
        with self.assertRaises(TypeError):
            SeedlessProcrustes(iterative_num_reps=3.14)
        with self.assertRaises(TypeError):
            SeedlessProcrustes(init=["hi", "there"])

        # value errors for all but initial Q and initial P
        with self.assertRaises(ValueError):
            SeedlessProcrustes(optimal_transport_lambda=-0.01)
        with self.assertRaises(ValueError):
            SeedlessProcrustes(optimal_transport_eps=-0.01)
        with self.assertRaises(ValueError):
            SeedlessProcrustes(optimal_transport_num_reps=-1)
        with self.assertRaises(ValueError):
            SeedlessProcrustes(iterative_num_reps=-1)
        with self.assertRaises(ValueError):
            SeedlessProcrustes(init="hi")

        # initial Q and initial P passed when shouldn't be
        with self.assertRaises(ValueError):
            SeedlessProcrustes(
                init="2d",
                initial_Q=np.eye(2),
            )
        with self.assertRaises(ValueError):
            SeedlessProcrustes(init="2d", initial_P=np.ones((100, 100)) / 10000)
        with self.assertRaises(ValueError):
            SeedlessProcrustes(
                init="custom",
                initial_Q=np.eye(2),
                initial_P=np.ones((100, 100)) / 10000,
            )

        # initial Q and initial P specific values
        # pass bad types
        with self.assertRaises(TypeError):
            SeedlessProcrustes(init="custom", initial_Q="hello there")
        with self.assertRaises(TypeError):
            SeedlessProcrustes(init="custom", initial_P="hello there")
        with self.assertRaises(TypeError):
            SeedlessProcrustes(init="custom", initial_Q={"hello": "there"})
        with self.assertRaises(TypeError):
            SeedlessProcrustes(init="custom", initial_P={"hello": "there"})
        # pass non ndim=2 matrices (cuaght by check_array)
        with self.assertRaises(ValueError):
            SeedlessProcrustes(init="custom", initial_Q=np.eye(5).reshape(5, 5, 1))
        with self.assertRaises(ValueError):
            SeedlessProcrustes(init="custom", initial_P=np.eye(5).reshape(5, 5, 1))
        # pass not an orthogonal matrix as a Q
        with self.assertRaises(ValueError):
            SeedlessProcrustes(init="custom", initial_Q=np.eye(3)[:-1])
        with self.assertRaises(ValueError):
            SeedlessProcrustes(init="custom", initial_Q=np.ones((3, 3)))
        # pass not a soft assignment matrix as P
        with self.assertRaises(ValueError):
            SeedlessProcrustes(init="custom", initial_P=np.ones((3, 2)))

    def test_bad_datasets(self):
        X = np.arange(6).reshape(6, 1)
        Y = np.arange(6).reshape(6, 1)
        Y_wrong_d = np.arange(12).reshape(6, 2)
        # check passing weird stuff as input (caught by us)
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes()
            aligner.fit("hello there", Y)
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes()
            aligner.fit(X, "hello there")
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes()
            aligner.fit({"hello": "there"}, Y)
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes()
            aligner.fit(X, {"hello": "there"})
        # check passing arrays of weird ndims (caught by check_array)
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes()
            aligner.fit(X, Y.reshape(3, 2, 1))
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes()
            aligner.fit(X.reshape(3, 2, 1), Y)
        # check passing arrays with different dimensions (caught by us)
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes()
            aligner.fit(X, Y_wrong_d)
        # check passing array with wrong dimensions to transform (caught by us)
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes()
            aligner.fit(X, Y)
            aligner.transform(Y_wrong_d)

    def test_different_inits(self):
        np.random.seed(314)
        mean = np.ones(3) * 5
        cov = np.eye(3) * 0.1
        X = stats.multivariate_normal.rvs(mean, cov, 100)
        print(X.shape)
        Y = stats.multivariate_normal.rvs(mean, cov, 100)
        W = stats.ortho_group.rvs(3)
        Y = Y @ W

        aligner_1 = SeedlessProcrustes(init="2d")
        aligner_1.fit(X, Y)

        aligner_2 = SeedlessProcrustes(init="sign_flips")
        aligner_2.fit(X, Y)
        test_sign_flips = SignFlips()
        self.assertTrue(
            np.all(test_sign_flips.fit(X, Y).Q_ == aligner_2.selected_initial_Q_)
        )

        aligner_3 = SeedlessProcrustes(init="custom")
        aligner_3.fit(X, Y)
        self.assertTrue(np.all(np.eye(3) == aligner_3.selected_initial_Q_))

        aligner_4 = SeedlessProcrustes(init="custom", initial_Q=-np.eye(3))
        aligner_4.fit(X, Y)
        self.assertTrue(np.all(-np.eye(3) == aligner_4.selected_initial_Q_))

        aligner_5 = SeedlessProcrustes(
            init="custom", initial_P=np.ones((100, 100)) / 10000
        )
        aligner_5.fit(X, Y)

    def test_aligning_datasets(self):
        np.random.seed(314)
        n, d = 250, 2
        mean = np.ones(d) * 5
        cov = np.ones((d, d)) * 0.02 + np.eye(d) * 0.8
        X = stats.multivariate_normal.rvs(mean, cov, n)
        Y = np.concatenate([X, X])
        W = stats.ortho_group.rvs(d)
        Y = Y @ W

        aligner = SeedlessProcrustes(init="2d")
        Q = aligner.fit(X, Y).Q_
        self.assertTrue(np.linalg.norm(Y.mean(axis=0) - (X @ Q).mean(axis=0)) < 0.1)


if __name__ == "__main__":
    unittest.main()
