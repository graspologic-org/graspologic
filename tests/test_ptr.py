import unittest
import numpy as np
from graspy.utils import pass_to_ranks, is_unweighted

class TestPTR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # simple ERxN graph
        n = 15
        p = 0.5
        cls.A = np.zeros((n, n))
        nedge = int(round(n * n * p))
        np.put(
            cls.A,
            np.random.choice(np.arange(0, n * n), size=nedge, replace=False),
            np.random.normal(size=nedge))

        cls.loopless_undirected_input = np.array([[0, 1, 60, 60], 
                                                [1, 0, 400, 0],
                                                [60, 400, 0, 80],
                                                [60, 0, 80, 0]])

        cls.looped_undirected_input = np.array([[0.5, 1, 60, 60], 
                                                [1, 0, 400, 0],
                                                [60, 400, 20, 80],
                                                [60, 0, 80, 0]])
    
    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            pass_to_ranks(self.A, method='hazelnutcoffe')
        with self.assertRaises(NotImplementedError):
            A = self.A
            A[2,0] = 1000 # make A asymmetric 
            pass_to_ranks(A)

    def test_zeroboost_loopless_undirected(self): 
        ptr_expected = np.array([[0, 2.0/6, 3.5/6, 3.5/6],
                                 [2.0/6, 0, 1, 0],
                                 [3.5/6, 1, 0, 5.0/6],
                                 [3.5/6, 0, 5.0/6, 0]])
        
        _run_test(self, 'zero-boost', self.loopless_undirected_input, ptr_expected)

    def test_zeroboost_looped_undirected(self): 
        ptr_expected = np.array([[0.4, 0.5, .75, 0.75],
                                [0.5, 0, 1, 0],
                                [0.75, 1, 0.6, 0.9],
                                [0.75, 0, 0.9, 0]])
        
        _run_test(self, 'zero-boost', self.looped_undirected_input, ptr_expected)

    def test_simpleall_loopless_undirected(self): 
        ptr_expected = np.array([[0, 0.1764706, 0.5294118, 0.5294118],
                                 [0.1764706, 0, 1.1176471, 0],
                                 [0.5294118, 1.1176471, 0, 0.8823529],
                                 [0.5294118, 0, 0.8823529, 0]])
        
        _run_test(self, 'simple-all', self.loopless_undirected_input, ptr_expected)

    def test_simpleall_looped_undirected(self): 
        ptr_expected = np.array([[0.1176471, 0.2941176, 0.7647059, 0.7647059],
                                 [0.2941176, 0, 1.3529412, 0],
                                 [0.7647059, 1.3529412, 0.4705882, 1.1176471],
                                 [0.7647059, 0, 1.1176471, 0]])

        _run_test(self, 'simple-all', self.looped_undirected_input, ptr_expected)

    def test_simplenonzero_loopless_undirected(self): 
        ptr_expected = np.array([[0, 0.2727273, 0.8181818, 0.8181818],
                                 [0.2727273, 0, 1.7272727, 0],
                                 [0.8181818, 1.7272727, 0, 1.3636364],
                                 [0.8181818, 0, 1.3636364, 0]])
        
        _run_test(self, 'simple-nonzero', self.loopless_undirected_input, ptr_expected)

    def test_simplenonzero_looped_undirected(self): 
        ptr_expected = np.array([[0.1538462, 0.3846154, 1, 1],
                                 [0.3846154, 0, 1.7692308, 0],
                                 [1, 1.7692308, 0.6153846, 1.4615385],
                                 [1, 0, 1.4615385, 0]])

        _run_test(self, 'simple-nonzero', self.looped_undirected_input, ptr_expected)
    


def _run_test(self, methodstr, input_graph, expected):
    ptr_out = pass_to_ranks(input_graph, method=methodstr)
    self.assertTrue(np.allclose(ptr_out, expected, rtol=1e-04))

if __name__ == '__main__':
    unittest.main()
