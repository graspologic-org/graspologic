# Bijan Varjavand
# bvarjav1 [at] jhu.edu
# 12.11.2018

import pytest
import numpy as np
from graspy.inference import NonparametricTest
from graspy.embed import AdjacencySpectralEmbed
from graspy.simulations import er_np

def gen(n=20):
    A1 = er_np(20, .3)
    A2 = er_np(n, .3)
    return A1, A2

def test_fit_p():
    A1, A2 = gen()
    p = NonparametricTest().fit(A1, A2)
    # TODO : something

def test_fit_asymmetric_p():
    A1, A2 = gen(30)
    p = NonparametricTest().fit(A1, A2)
    # TODO : something

def test_bad_kwargs():
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

def test_n_bootstraps():
    A1, A2 = gen()
    npt = NonparametricTest(n_bootstraps=234)
    npt.fit(A1, A2)
    assert len(npt.U_bootstrap) == 234

def test_bad_matrix_inputs():
    npt = NonparametricTest()
    A1, A2 = gen()
    # svd indexerror?
