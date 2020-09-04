# Anton Alyakin
# aalyaki1 [at] jhu.edu
# 09.04.2020

import unittest

import numpy as np
from scipy import stats
from graspy.align import SeedlessProcrustes


class TestSeedlessProcrustes(unittest.TestCase):
    def test_bad_kwargs(self):
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
        with self.assertRaises(ValueError):
            SeedlessProcrustes(initialization="hi")

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
