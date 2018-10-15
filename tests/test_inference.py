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
        print(p)

        