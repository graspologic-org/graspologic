import unittest
import numpy as np
from graspy.inference import SemiparametricTest

class TestSemiparametricTest(unittest.TestCase):

    def test_private_embed(self):
        '''
            throwaway, temporary test
        '''
        A1 = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])

        A2 = np.array([[1, 1, 1],
                       [1, 1, 1],
                       [1, 1, 1]])

        spt = SemiparametricTest()
        X1_hat, X2_hat = spt._embed(A1, A2)

    def test_private_bootstraps(self):
        '''
            throwaway, temporary test
        '''
        A1 = np.array([[0, 1, 0],
                       [1, 0, 1],
                       [0, 1, 0]])

        A2 = np.array([[1000, 1, 0],
                       [1 ,1, 1],
                       [0, 1, 1000]])

        spt = SemiparametricTest()
        X_hats = spt._embed(A1, A2)
        ts = spt._bootstrap(X_hats)
        print(ts)