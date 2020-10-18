# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import warnings

import numpy as np

np.random.seed(8888)
from scipy import stats

from ..embed import select_dimension, AdjacencySpectralEmbed
from ..utils import import_graph, fit_plug_in_variance_estimator
from ..align import SignFlips
from ..align import SeedlessProcrustes
from .base import BaseInference
from sklearn.utils import check_array
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics.pairwise import PAIRED_DISTANCES
from sklearn.metrics.pairwise import PAIRWISE_KERNEL_FUNCTIONS
from hyppo.ksample import KSample
from hyppo._utils import gaussian

_VALID_DISTANCES = list(PAIRED_DISTANCES.keys())
_VALID_KERNELS = list(PAIRWISE_KERNEL_FUNCTIONS.keys())
_VALID_KERNELS.append("gaussian")  # can use hyppo's medial gaussian kernel too
_VALID_METRICS = _VALID_DISTANCES + _VALID_KERNELS

_VALID_TESTS = ["cca", "dcorr", "hhg", "rv", "hsic", "mgc"]


def ldt_function(
    A1,
    A2,
    test="dcorr",
    metric="euclidean",
    n_components=None,
    n_bootstraps=200,
    workers=1,
    size_correction=True,
    pooled=False,
    align_type="sign_flips",
    align_kws={},
    input_graph=True,
):

    # check test argument
    if not isinstance(test, str):
        msg = "test must be a str, not {}".format(type(test))
        raise TypeError(msg)
    elif test not in _VALID_TESTS:
        msg = "Unknown test {}. Valid tests are {}".format(test, _VALID_TESTS)
        raise ValueError(msg)
    # metric argument is checked when metric_func_ is instantiated
    # check n_components argument
    if n_components is not None:
        if not isinstance(n_components, int):
            msg = "n_components must be an int, not {}.".format(type(n_components))
            raise TypeError(msg)
    # check n_bootstraps argument
    if not isinstance(n_bootstraps, int):
        msg = "n_bootstraps must be an int, not {}".format(type(n_bootstraps))
        raise TypeError(msg)
    elif n_bootstraps < 0:
        msg = "{} is invalid number of bootstraps, must be non-negative"
        raise ValueError(msg.format(n_bootstraps))
    # check workers argument
    if not isinstance(workers, int):
        msg = "workers must be an int, not {}".format(type(workers))
        raise TypeError(msg)
    # check size_correction argument
    if not isinstance(size_correction, bool):
        msg = "size_correction must be a bool, not {}".format(type(size_correction))
        raise TypeError(msg)
    # check pooled argument
    if not isinstance(pooled, bool):
        msg = "pooled must be a bool, not {}".format(type(pooled))
        raise TypeError(msg)
    # check align_type argument
    if (not isinstance(align_type, str)) and (align_type is not None):
        msg = "align_type must be a string or None, not {}".format(type(align_type))
        raise TypeError(msg)
    align_types_supported = ["sign_flips", "seedless_procrustes", None]
    if align_type not in align_types_supported:
        msg = "supported align types are {}".format(align_types_supported)
        raise ValueError(msg)
    # check align_kws argument
    if not isinstance(align_kws, dict):
        msg = "align_kws must be a dictionary of keyword arguments, not {}".format(
            type(align_kws)
        )
        raise TypeError(msg)
    # check input_graph argument
    if not isinstance(input_graph, bool):
        msg = "input_graph must be a bool, not {}".format(type(input_graph))
        raise TypeError(msg)

    if input_graph:
        A1 = import_graph(A1)
        A2 = import_graph(A2)

        X1_hat, X2_hat = _embed(A1, A2, n_components)
    else:
        # check for nx objects, since they are castable to arrays,
        # but we don't want that
        if not isinstance(A1, np.ndarray):
            msg = (
                f"Embedding of the first graph is of type {type(A1)}, not "
                "np.ndarray. If input_graph is False, the inputs need to be "
                "adjacency spectral embeddings, with shapes (n, d) and "
                "(m, d), passed as np.ndarrays."
            )
            raise TypeError(msg)
        if not isinstance(A2, np.ndarray):
            msg = (
                f"Embedding of the second graph is of type {type(A2)}, not an "
                "array. If input_graph is False, the inputs need to be "
                "adjacency spectral embeddings, with shapes (n, d) and "
                "(m, d), passed as np.ndarrays."
            )
            raise TypeError(msg)

        if A1.ndim != 2:
            msg = (
                "Embedding array of the first graph does not have two dimensions. "
                "If input_graph is False, the inputs need to be adjacency "
                "spectral embeddings, with shapes (n, d) and (m, d)"
            )
            raise ValueError(msg)
        if A2.ndim != 2:
            msg = (
                "Embedding array of the second graph does not have two dimensions. "
                "If input_graph is False, the inputs need to be adjacency "
                "spectral embeddings, with shapes (n, d) and (m, d)"
            )
            raise ValueError(msg)
        if A1.shape[1] != A2.shape[1]:
            msg = (
                "Two input embeddings have different number of components. "
                "If input_graph is False, the inputs need to be adjacency "
                "spectral embeddings, with shapes (n, d) and (m, d)"
            )
            raise ValueError(msg)

        # checking for inf values
        X1_hat = check_array(A1)
        X2_hat = check_array(A2)

    if align_type == "sign_flips":
        aligner = SignFlips(**align_kws)
        X1_hat = aligner.fit_transform(X1_hat, X2_hat)
    elif align_type == "seedless_procrustes":
        aligner = SeedlessProcrustes(**align_kws)
        X1_hat = aligner.fit_transform(X1_hat, X2_hat)

    if size_correction:
        X1_hat, X2_hat = _sample_modified_ase(X1_hat, X2_hat, pooled=pooled)

    metric_func_ = _instantiate_metric_func(metric=metrix, test=test)
    test_obj = KSample(test, compute_distance=metric_func_)

    data = test_obj.test(X1_hat, X2_hat, reps=n_bootstraps, workers=workers, auto=False)

    null_distribution_ = test_obj.indep_test.null_dist
    sample_T_statistic_ = data[0]
    p_value_ = data[1]

    return p_value_, sample_T_statistic, null_distribution_


def _instantiate_metric_func(metric, test):
    # check metric argument
    if not isinstance(metric, str) and not callable(metric):
        msg = "Metric must be str or callable, not {}".format(type(metric))
        raise TypeError(msg)
    elif metric not in _VALID_METRICS and not callable(metric):
        msg = "Unknown metric {}. Valid metrics are {}, or a callable".format(
            metric, _VALID_METRICS
        )
        raise ValueError(msg)
    if callable(metric):
        metric_func = metric
    else:
        if metric in _VALID_DISTANCES:
            if test == "hsic":
                msg = (
                    f"{test} is a kernel-based test, but {metric} "
                    "is a distance. results may not be optimal. it is "
                    "recomended to use either a different test or one of "
                    f"the kernels: {_VALID_KERNELS} as a metric."
                )
                warnings.warn(msg, UserWarning)

            def metric_func(X, Y=None, metric=metric, workers=None):
                return pairwise_distances(X, Y, metric=metric, n_jobs=workers)

        elif metric == "gaussian":
            if test != "hsic":
                msg = (
                    f"{test} is a distance-based test, but {metric} "
                    "is a kernel. results may not be optimal. it is "
                    "recomended to use either a hisc as a test or one of "
                    f"the distances: {_VALID_DISTANCES} as a metric."
                )
                warnings.warn(msg, UserWarning)
            metric_func = gaussian
        else:
            if test != "hsic":
                msg = (
                    f"{test} is a distance-based test, but {metric} "
                    "is a kernel. results may not be optimal. it is "
                    "recomended to use either a hisc as a test or one of "
                    f"the distances: {_VALID_DISTANCES} as a metric."
                )
                warnings.warn(msg, UserWarning)

            def metric_func(X, Y=None, metric=metric, workers=None):
                return pairwise_kernels(X, Y, metric=metric, n_jobs=workers)

    return metric_func


def _embed(A1, A2, n_components):
    if n_components is None:
        num_dims1 = select_dimension(A1)[0][-1]
        num_dims2 = select_dimension(A2)[0][-1]
        n_components = max(num_dims1, num_dims2)

    ase = AdjacencySpectralEmbed(n_components=n_components)
    X1_hat = ase.fit_transform(A1)
    X2_hat = ase.fit_transform(A2)

    if isinstance(X1_hat, tuple) and isinstance(X2_hat, tuple):
        X1_hat = np.concatenate(X1_hat, axis=-1)
        X2_hat = np.concatenate(X2_hat, axis=-1)
    elif isinstance(X1_hat, tuple) ^ isinstance(X2_hat, tuple):
        msg = (
            "input graphs do not have same directedness. "
            "consider symmetrizing the directed graph."
        )
        raise ValueError(msg)

    return X1_hat, X2_hat


def _sample_modified_ase(X, Y, pooled=False):
    N, M = len(X), len(Y)

    # return if graphs are same order, else else ensure X the larger graph.
    if N == M:
        return X, Y
    elif M > N:
        reverse_order = True
        X, Y = Y, X
        N, M = M, N
    else:
        reverse_order = False

    # estimate the central limit theorem variance
    if pooled:
        two_samples = np.concatenate([X, Y], axis=0)
        get_sigma = fit_plug_in_variance_estimator(two_samples)
    else:
        get_sigma = fit_plug_in_variance_estimator(X)
    X_sigmas = get_sigma(X) * (N - M) / (N * M)

    # increase the variance of X by sampling from the asy dist
    X_sampled = np.zeros(X.shape)
    # TODO may be parallelized, but requires keeping track of random state
    for i in range(N):
        X_sampled[i, :] = X[i, :] + stats.multivariate_normal.rvs(cov=X_sigmas[i])

    # return the embeddings in the appropriate order
    return (Y, X_sampled) if reverse_order else (X_sampled, Y)
