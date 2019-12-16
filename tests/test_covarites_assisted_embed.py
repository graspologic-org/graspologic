# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 20:13:45 2019

@author: jerryyao
"""

import pytest
import graspy as gs
import numpy as np

from graspy.embed.casc import CovariateAssistedSpectralEmbed
from graspy.embed.lse import LaplacianSpectralEmbed
from graspy.embed.svd import selectSVD
from graspy.simulations.simulations import er_np, er_nm, sbm
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score as ARI

"""
All these tests uses :
A SBM model generated adjacency matrix with very few nodes and significant different block probability
A covarites matrix where covarite probability is significantlly differently different between blocks.
If casc is correctly installed , it will perform 100% correct clustering on this case.
"""


def test_casc_cca():

    n = [10, 10]
    p = [[0.8, 0.2], [0.2, 0.8]]
    np.random.seed(105)
    A = sbm(n=n, p=p)
    covarites = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    casc = CovariateAssistedSpectralEmbed(
        n_components=2, assortative=True, cca=True, check_lcc=False
    )
    casc_results = casc.fit(np.array(A), covarites)
    results_ans = {
        "cluster": ([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    }
    ResultARI = ARI(casc_results["cluster"], results_ans["cluster"])

    assert ResultARI == 1


def test_casc_assort():
    n = [10, 10]
    p = [[0.8, 0.2], [0.2, 0.8]]
    np.random.seed(105)
    A = sbm(n=n, p=p)
    covarites = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    casc = CovariateAssistedSpectralEmbed(
        n_components=2, assortative=True, cca=False, check_lcc=False
    )
    casc_results = casc.fit(np.array(A), covarites)

    results_ans = {
        "cluster": ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    }
    ResultARI = ARI(casc_results["cluster"], results_ans["cluster"])
    assert ResultARI == 1


def test_casc_non_assort():
    n = [10, 10]
    p = [[0.8, 0.2], [0.2, 0.8]]
    np.random.seed(105)
    A = sbm(n=n, p=p)
    covarites = np.array(
        [
            [1.0, 0.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    casc = CovariateAssistedSpectralEmbed(
        n_components=2, assortative=False, cca=False, check_lcc=False
    )
    casc_results = casc.fit(np.array(A), covarites)

    results_ans = {
        "cluster": ([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]),
    }
    ResultARI = ARI(casc_results["cluster"], results_ans["cluster"])
    assert ResultARI == 1
