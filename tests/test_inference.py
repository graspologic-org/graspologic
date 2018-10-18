import unittest
import numpy as np
from graspy.inference import SemiparametricTest
from graspy.embed import AdjacencySpectralEmbed

class TestSemiparametricTest(unittest.TestCase):

    def test_fit_p(self):
        A1 = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])

        A2 = np.array([[0, 1, 0],
                       [1 ,0, 1],
                       [0, 1, 0]])

        spt = SemiparametricTest()
        p = spt.fit(A1, A2)

    def test_input_checking(self):
        with self.assertRaises(ValueError):
            spt = SemiparametricTest(n_components=-100)
        with self.assertRaises(ValueError):
            SemiparametricTest(n_components=-100)
        with self.assertRaises(ValueError):
            spt = SemiparametricTest(test_case='oops')
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

if __name__ == '__main__':
    unittest.main()