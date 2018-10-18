# Ben Pedigo 
# bpedigo [at] jhu.edu
# 10.18.2018

import unittest
import numpy as np
from graspy.inference import SemiparametricTest
from graspy.embed import AdjacencySpectralEmbed


class TestSemiparametricTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.A1 = np.array([[0, 1, 0],
                           [1, 0, 1],
                           [0, 1, 0]])

        cls.A2 = np.array([[0, 1, 0],
                           [1 ,0, 1],
                           [0, 1, 0]])

    def test_fit_p(self):
        spt = SemiparametricTest()
        p = spt.fit(self.A1, self.A2)

    def test_bad_kwargs(self):
        with self.assertRaises(ValueError):
            SemiparametricTest(n_components=-100)
        with self.assertRaises(ValueError):
            SemiparametricTest(n_components=-100)
        with self.assertRaises(ValueError):
            SemiparametricTest(test_case='oops')
        with self.assertRaises(ValueError):
            SemiparametricTest(n_bootstraps=-100)
        with self.assertRaises(ValueError):
            SemiparametricTest(embedding='oops')
        with self.assertRaises(TypeError):
            SemiparametricTest(n_bootstraps=0.5)
        with self.assertRaises(TypeError):
            SemiparametricTest(n_components=0.5)
        with self.assertRaises(TypeError):
            SemiparametricTest(embedding=6)
        with self.assertRaises(TypeError):
            SemiparametricTest(test_case=6)

    def test_n_bootstraps(self):
        spt = SemiparametricTest(n_bootstraps=234)
        spt.fit(self.A1, self.A2)
        self.assertEqual(spt.T1_bootstrap.shape[0], 234)
    
    def test_bad_matrix_inputs(self):
        spt = SemiparametricTest()
        A1 = self.A1.copy()
        A1[2,0] = 1 # make asymmetric
        with self.assertRaises(NotImplementedError): # TODO : remove when we implement
            spt.fit(A1, self.A2)

        bad_matrix = [[1, 2]]
        with self.assertRaises(TypeError):
            spt.fit(bad_matrix, self.A2)

        with self.assertRaises(ValueError):
            spt.fit(self.A1[:2,:2], self.A2)

if __name__ == '__main__':
    unittest.main()