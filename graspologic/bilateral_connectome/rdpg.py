import numpy as np
from graspologic.embed import select_dimension, AdjacencySpectralEmbed
from graspologic.utils import augment_diagonal
from giskard.align import joint_procrustes
from sklearn.preprocessing import normalize
from hyppo.ksample import KSample
import time


def embed(adj, n_components=40):
    elbow_inds, _ = select_dimension(augment_diagonal(adj), n_elbows=5)
    elbow_inds = np.array(elbow_inds)
    ase = AdjacencySpectralEmbed(
        n_components=n_components, check_lcc=False, diag_aug=True, concat=False
    )
    out_latent, in_latent = ase.fit_transform(adj)
    return out_latent, in_latent, ase.singular_values_, elbow_inds


def rdpg_test(
    A1,
    A2,
    n_components=8,
    align_n_components=None,
    seeds=None,
    normalize_nodes=False,
):
    if align_n_components is None:
        align_n_components = n_components

    misc = {}

    # Xs are the so called "out" latent positions
    # Ys are the "in" latent positions
    currtime = time.time()
    X1, Y1, singular_values1, elbow_inds1 = embed(A1, n_components=align_n_components)
    X2, Y2, singular_values2, elbow_inds2 = embed(A2, n_components=align_n_components)
    misc["embed_time"] = time.time() - currtime
    misc["singular_values1"] = singular_values1
    misc["singular_values2"] = singular_values2
    # misc["elbow_inds1"] = elbow_inds1
    # misc["elbow_inds2"] = elbow_inds2

    if normalize_nodes:
        # projects latent positions for each node to the unit sphere
        # TODO should this be concatenated out/in first?
        X1 = normalize(X1)
        Y1 = normalize(Y1)
        X2 = normalize(X2)
        Y2 = normalize(Y2)

    currtime = time.time()
    X1, Y1 = joint_procrustes(
        (X1, Y1),
        (X2, Y2),
        method="transport",
        seeds=seeds,
    )
    misc["align_time"] = time.time() - currtime

    Z1 = np.concatenate((X1[:, :n_components], Y1[:, :n_components]), axis=1)
    Z2 = np.concatenate((X2[:, :n_components], Y2[:, :n_components]), axis=1)

    misc["Z1"] = Z1
    misc["Z2"] = Z2

    currtime = time.time()
    test_obj = KSample("dcorr")
    stat, pvalue = test_obj.test(
        Z1,
        Z2,
        reps=1000,
        workers=-1,
        auto=True,
    )
    misc["test_time"] = time.time() - currtime

    return stat, pvalue, misc
