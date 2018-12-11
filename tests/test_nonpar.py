# Bijan Varjavand
# bvarjav1 [at] jhu.edu
# 12.11.2018

import pytest
import numpy as np
from graspy.inference import NonparametricTest
from graspy.embed import AdjacencySpectralEmbed
from graspy.simulations import er_np

def gen():
    A1 = er_np(20, .3)
    A2 = er_np(20, .3)
    return A1, A2

def test_fit_p(A1, A2):
    p = NonparametricTest().fit(gen())
    # TODO : something

def test_bad_kwargs(self):
    with pytest.raises(ValueError):
        NonparametricTest(n_components=-100)
    with pytest.raises(ValueError):
        NonparametricTest(n_components=-100)
    with pytest.raises(ValueError):
        NonparametricTest(n_bootstraps=-100)
    with pytest.raises(ValueError):
        NonparametricTest(embedding='oops')
    with pytest.raises(TypeError):
        NonparametricTest(n_bootstraps=0.5)
    with pytest.raises(TypeError):
        NonparametricTest(n_components=0.5)
    with pytest.raises(TypeError):
        NonparametricTest(embedding=6)
    with pytest.raises(TypeError):
        NonparametricTest(test_case=6)

def test_n_bootstraps(self):
    spt = NonparametricTest(n_bootstraps=234)
    spt.fit(gen())
    self.assertEqual(spt.U_bootstrap.shape[0], 234)

def test_bad_matrix_inputs(self):
    npt = NonparametricTest()
    A1, A2 = gen()

    bad_matrix = [[1, 2]]
    with self.assertRaises(TypeError):
        npt.fit(bad_matrix, A2)

    with self.assertRaises(ValueError):
        npt.fit(A1[:2,:2], A2)
