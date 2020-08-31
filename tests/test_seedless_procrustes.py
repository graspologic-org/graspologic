# Anton Alyakin
# aalyaki1 [at] jhu.edu
# 09.01.2020

import unittest

import numpy as np
from scipy import stats
from graspy.align import SeedlessProcrustes


class TestSeedlessProcrustes(unittest.TestCase):
    def test_bad_kwargs(self):
        pass
        # # ensure that i am checking all possible kwargs
        # with self.assertRaises(TypeError):
        #     SeedlessProcrustes(lambda_init="oops")
        # with self.assertRaises(TypeError):
        #     SeedlessProcrustes(lambda_final="oops")
        # with self.assertRaises(TypeError):
        #     SeedlessProcrustes(alpha="oops")
        # with self.assertRaises(TypeError):
        #     SeedlessProcrustes(optimal_transport_eps="oops")
        # with self.assertRaises(TypeError):
        #     SeedlessProcrustes(iterative_eps="oops")
        # with self.assertRaises(TypeError):
        #     SeedlessProcrustes(num_reps=3.15)
        # with self.assertRaises(ValueError):
        #     SeedlessProcrustes(alpha=-0.01)
        # with self.assertRaises(ValueError):
        #     SeedlessProcrustes(optimal_transport_eps=-0.01)
        # with self.assertRaises(ValueError):
        #     SeedlessProcrustes(iterative_eps=-0.01)
        # with self.assertRaises(ValueError):
        #     SeedlessProcrustes(num_reps=0)

    def test_matching_datasets(self):
        np.random.seed(314)
        X = np.random.normal(1, 0.2, 1000).reshape(-1, 4)
        Y = np.concatenate([X, X])
        W = stats.ortho_group.rvs(4)
        Y = Y @ W

        aligner = SeedlessProcrustes(initialization="custom", initial_Q=np.eye(4))
        Q = aligner.fit(X, Y).Q_X
        self.assertTrue(np.all(np.isclose(Q, W)))


if __name__ == "__main__":
    unittest.main()
