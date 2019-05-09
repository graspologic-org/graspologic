import unittest
import warnings
import numpy as np
from graspy.simulations import lsm
from graspy.inference import LatentStructureTest

class TestLatentStructureTest(unittest.TestCase):

    def test_lsm_generats(self):
        null_density_fn = lambda : np.random.beta(1, 3)
        curve_fn = lambda t: [np.sin(t * np.pi), np.cos(t * np.pi), 0]
        A = lsm(null_density_fn, curve_fn, 100)
        self.assertEqual(A.shape, (100, 100))

        null_density_fn = lambda : np.random.beta(1, 3)
        curve_fn = lambda t: [np.sin(t * np.pi), np.cos(t * np.pi), 0]
        A = lsm(null_density_fn, curve_fn, 50)
        self.assertEqual(A.shape, (50, 50))

    def test_lsm_stat(self):
        null_density_fn = lambda : np.random.beta(1, 3)
        alt_density_fn = lambda : np.random.beta(1, 1)
        curve_fn = lambda t: [np.sin(t * np.pi), np.cos(t * np.pi), 0]
        A0_0 = lsm(null_density_fn, curve_fn, 512)
        A0_1 = lsm(null_density_fn, curve_fn, 512)
        A1 = lsm(alt_density_fn, curve_fn, 512)
        lst = LatentStructureTest(3)
        T_diff = lst.fit(A0_0, A1)['stat']
        T_same = lst.fit(A0_0, A0_1)['stat']
        self.assertTrue(T_diff > T_same)

    def test_neighbors_warns(self):
        null_density_fn = lambda : np.random.beta(1, 3)
        curve_fn = lambda t: [np.sin(t * np.pi), np.cos(t * np.pi), 0]
        A0 = lsm(null_density_fn, curve_fn, 512)
        A1 = lsm(null_density_fn, curve_fn, 512)
        lst = LatentStructureTest(3)
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            lst.fit(A0, A1, initial_neighbors=50, use_min=False)
        #raises a warning for each adj matrix
        assert(len(w)==2)
