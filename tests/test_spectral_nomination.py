import pytest
import numpy as np
from typing import List, Tuple

from graspologic.simulations.simulations import er_np, er_nm, sbm
from graspologic.nominate import SpectralVertexNominator
from graspologic.nominate import SpectralClusterVertexNominator

def _unattributed(seed, n_verts):
    p = [[.4, .2],
         [.2, .4]]
    adj = sbm(2 * [n_verts], p)
    svn = SpectralVertexNominator(mode='single_vertex')
    svn.fit(adj, seed)
    nom_list, dists, att_map = svn.predict()
    # some basic assertions on output shape here
    assert nom_list.shape == (2*n_verts, seed.shape[0])
    assert dists.shape == (n_verts, seed.shape[0])
    assert att_map.shape == tuple([1]) and att_map[0] == 0
    return nom_list


def _attributed(seed_size, n_verts, nominator):
    p = np.array([[0.1, .3, .4],
                  [.3, 0.5, .6],
                  [.4, .6, .25]])
    adj = sbm(3*n_verts, p)
    labels = [0]*n_verts + [1]*n_verts + [2]*n_verts
    seed = np.empty((seed_size, 2))
    seed[:, 0] = np.random.choice(3*n_verts, size=seed_size, replace=False)
    seed[:, 1] = labels[seed[:, 0]]
    nominator.fit(adj, seed)
    nom_list, dists, att_map = nominator.predict(out='per_attribute')
    unique_att = nominator.unique_att
    assert nom_list.shape == (3*n_verts, unique_att.shape[0])
    assert dists.shape == (3*n_verts, unique_att.shape[0])
    assert att_map.shape == unique_att.shape
    return nom_list


def test_unattributed_basic():
    _unattributed(np.array([8]), 10)
    _unattributed(np.array([2, 6, 9, 15, 25]), 15)

@pytest.mark.parameterize("nominator",
                          [SpectralVertexNominator(),
                           SpectralClusterVertexNominator()])
def test_attributed_basic(nominator):
    _attributed(50, 100, nominator)
    _attributed(20, 1000, nominator)

@pytest.mark.parameterize("nominator",
                          [SpectralVertexNominator(),
                           SpectralClusterVertexNominator()])
def test_attributed_acc(nominator):
    """
    want to ensure preformace is at least better than random
    """
    group1_correct = np.zeros(300)
    group2_correct = np.zeros(300)
    group3_correct = np.zeros(300)
    labels = [0] * 100 + [1] * 100 + [2] * 100
    for i in range(100):
        n_list = _attributed(seed_size=50, n_verts=300, nominator=nominator)
        group1_correct[np.argwhere(labels[n_list.T[0]] == 0)] += 1
        group2_correct[np.argwhere(labels[n_list.T[1]] == 1)] += 1
        group3_correct[np.argwhere(labels[n_list.T[2]] == 2)] += 1
    g1_correct_prob = group1_correct / 100
    g2_correct_prob = group2_correct / 100
    g3_correct_prob = group3_correct / 100
    assert np.min(g1_correct_prob[:100]) > .5
    assert np.min(g2_correct_prob[:100]) > .5
    assert np.min(g2_correct_prob[:100]) > .5

# TODO: More test of various options, passing embedding, edge cases, warnings, type assertions