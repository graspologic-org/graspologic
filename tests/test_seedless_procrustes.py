# Anton Alyakin
# aalyaki1 [at] jhu.edu
# 09.04.2020

import unittest

import numpy as np
from scipy import stats
from graspy.align import SeedlessProcrustes


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
            SeedlessProcrustes(iterative_eps="oops")
        with self.assertRaises(TypeError):
            SeedlessProcrustes(iterative_num_reps=3.14)
        with self.assertRaises(TypeError):
            SeedlessProcrustes(initialization=["hi", "there"])

        # value errors for all but initial Q and initial P
        with self.assertRaises(ValueError):
            SeedlessProcrustes(optimal_transport_lambda=-0.01)
        with self.assertRaises(ValueError):
            SeedlessProcrustes(optimal_transport_eps=-0.01)
        with self.assertRaises(ValueError):
            SeedlessProcrustes(optimal_transport_num_reps=0)
        with self.assertRaises(ValueError):
            SeedlessProcrustes(iterative_eps=-0.01)
        with self.assertRaises(ValueError):
            SeedlessProcrustes(iterative_num_reps=0)
        with self.assertRaises(NotImplementedError):
            SeedlessProcrustes(initialization="hi")

        # initial Q and initial P things
        # pass bad types
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes(initial_Q="hello there")
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes(initial_P="hello there")
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes(initial_Q={"hello": "there"})
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes(initial_P={"hello": "there"})
        # pass non ndim=2 matrices (cuaght by check_array)
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes(initial_Q=np.ones(25).reshape(5, 5, 1))
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes(initial_P=np.ones(25).reshape(5, 5, 1))
        # pass not an orthogonal matrix as a Q
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes(initial_Q=np.ones((3, 2)))
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes(initial_Q=np.ones((3, 3)))
        # pass not a "doubly stochasitc" matrix as P
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes(initial_P=np.ones((3, 2)))
        # pass a correct doubly stochastic matrix as P
        SeedlessProcrustes(initial_P=np.ones((3, 2)) / 6)

    def test_bad_datasets(self):
        X = np.arange(6).reshape(6, 1)
        Y = np.arange(6).reshape(6, 1)
        Y_wrong_d = np.arange(12).reshape(6, 2)
        Y_wrong_n = np.arange(12).reshape(12, 1)
        # check passing weird stuff as input (caught by us)
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes()
            aligner.fit_transform("hello there", Y)
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes()
            aligner.fit_transform(X, "hello there")
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes()
            aligner.fit_transform({"hello": "there"}, Y)
        with self.assertRaises(TypeError):
            aligner = SeedlessProcrustes()
            aligner.fit_transform(X, {"hello": "there"})
        # check passing arrays of weird ndims (caught by check_array)
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes()
            aligner.fit_transform(X, Y.reshape(3, 2, 1))
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes()
            aligner.fit_transform(X.reshape(3, 2, 1), Y)
        # check passing arrays with different dimensions (caught by us)
        with self.assertRaises(ValueError):
            aligner = SeedlessProcrustes()
            aligner.fit_transform(X, Y_wrong_d)

    # def test_matching_datasets(self):
    #     np.random.seed(314)
    #     X = np.random.normal(1, 0.2, 1000).reshape(-1, 4)
    #     Y = np.concatenate([X, X])
    #     W = stats.ortho_group.rvs(4)
    #     Y = Y @ W

    #     aligner = SeedlessProcrustes(initialization="custom", initial_Q=np.eye(4))
    #     Q = aligner.fit(X, Y).Q_X
    #     self.assertTrue(np.all(np.isclose(Q, W)))


if __name__ == "__main__":
    unittest.main()
