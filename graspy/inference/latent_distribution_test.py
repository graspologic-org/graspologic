# Copyright 2020 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings

import numpy as np
from scipy import stats

from ..embed import select_dimension, AdjacencySpectralEmbed
from ..utils import import_graph, is_symmetric
from .base import BaseInference
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


class LatentDistributionTest(BaseInference):
    """
    Two-sample hypothesis test for the problem of determining whether two random
    dot product graphs have the same distributions of latent positions.

    This test can operate on two graphs where there is no known matching
    between the vertices of the two graphs, or even when the number of vertices
    is different. Currently, testing is only supported for undirected graphs.

    Read more in the :ref:`tutorials <inference_tutorials>`

    Parameters
    ----------
    test : str
        Backend hypothesis test to use, one of ["cca", "dcorr", "hhg", "rv", "hsic", "mgc"].
        These tests are typically used for independence testing, but here they
        are used for a two-sample hypothesis test on the latent positions of
        two graphs. See :class:`hyppo.ksample.KSample` for more information.

    metric : str or function, (default="gaussian")
        Distance or a kernel metric to use, either a callable or a valid string.
        If a callable, then it should behave similarly to either
        :func:`sklearn.metrics.pairwise_distances` or to
        :func:`sklearn.metrics.pairwise.pairwise_kernels`.
        If a string, then it should be either one of the keys in either
        `sklearn.metrics.pairwise.PAIRED_DISTANCES` or in
        `sklearn.metrics.pairwise.PAIRWISE_KERNEL_FUNCTIONS`, or "gaussian",
        which will use a gaussian kernel with an adaptively selected bandwidth.
        It is recommended to use kernels (e.g. "gaussian") with kernel-based
        hsic test and distances (e.g. "euclidean") with all other tests.

    n_components : int or None, optional (default=None)
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.
        See :func:`~graspy.embed.selectSVD` for more information.

    n_bootstraps : int (default=200)
        Number of bootstrap iterations for the backend hypothesis test.
        See :class:`hyppo.ksample.KSample` for more information.

    workers : int, optional (default=1)
        Number of workers to use. If more than 1, parallelizes the code.
        Supply -1 to use all cores available to the Process.

    order_correction: bool (default=True)
        The test degrades in validity as the orders of two graphs diverge from
        each other, unless the kernel matrix is modified.
        If True - in the case when two graphs are not of equal orders, estimates
        the plug-in estimator for the variance and uses it to correct the
        embedding of the larger graph.
        If False - does not perform any modifications (generally not
        recommended).

    Attributes
    ----------
    null_distribution_ : ndarray, shape (n_bootstraps, )
        The distribution of T statistics generated under the null.

    sample_T_statistic_ : float
        The observed difference between the embedded latent positions of the two
        input graphs.

    p_value_ : float
        The overall p value from the test.

    References
    ----------
    .. [1] Tang, M., Athreya, A., Sussman, D. L., Lyzinski, V., & Priebe, C. E. (2017).
        "A nonparametric two-sample hypothesis testing problem for random graphs."
        Bernoulli, 23(3), 1599-1630.

    .. [2] Panda, S., Palaniappan, S., Xiong, J., Bridgeford, E., Mehta, R., Shen, C., & Vogelstein, J. (2019).
        "hyppo: A Comprehensive Multivariate Hypothesis Testing Python Package."
        arXiv:1907.02088.

    .. [3] Varjavand, B., Arroyo, J., Tang, M., Priebe, C., and Vogelstein, J. (2019).
       "Improving Power of 2-Sample Random Graph Tests with Applications in Connectomics"
       arXiv:1911.02741

    .. [4] Alyakin, A., Agterberg, J., Helm, H., Priebe, C. (2020)
       "Correcting a Nonparametric Two-sample Graph Hypothesis test for Differing Orders"
       TODO cite the arXiv whenever possible
    """

    def __init__(
        self,
        test="dcorr",
        metric="euclidean",
        n_components=None,
        n_bootstraps=200,
        workers=1,
        order_correction=True,
    ):

        if not isinstance(test, str):
            msg = "test must be a str, not {}".format(type(test))
            raise TypeError(msg)
        elif test not in _VALID_TESTS:
            msg = "Unknown test {}. Valid tests are {}".format(test, _VALID_TESTS)
            raise ValueError(msg)

        if not isinstance(metric, str) and not callable(metric):
            msg = "Metric must be str or callable, not {}".format(type(metric))
            raise TypeError(msg)
        elif metric not in _VALID_METRICS and not callable(metric):
            msg = "Unknown metric {}. Valid metrics are {}, or a callable".format(
                metric, _VALID_METRICS
            )
            raise ValueError(msg)

        if n_components is not None:
            if not isinstance(n_components, int):
                msg = "n_components must be an int, not {}.".format(type(n_components))
                raise TypeError(msg)

        if not isinstance(n_bootstraps, int):
            msg = "n_bootstraps must be an int, not {}".format(type(n_bootstraps))
            raise TypeError(msg)
        elif n_bootstraps < 0:
            msg = "{} is invalid number of bootstraps, must be positive"
            raise ValueError(msg.format(n_bootstraps))

        if not isinstance(workers, int):
            msg = "workers must be an int, not {}".format(type(workers))
            raise TypeError(msg)
        elif workers == 0 or workers < 1:
            msg = (
                "{} is invalid number of workers, must be positive, or -1 (to use all)"
            )
            raise ValueError(msg.format(workers))

        if not isinstance(order_correction, bool):
            msg = "order_correction must be a bool, not {}".format(
                type(order_correction)
            )
            raise TypeError(msg)

        super().__init__(n_components=n_components)

        if callable(metric):
            metric_func = metric
        else:
            if metric in _VALID_DISTANCES:
                if test == "hsic":
                    msg = (
                        f"{test} is a kernel-baed test, but {metric} "
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
                        f"{test} is a distance-baed test, but {metric} "
                        "is a kernel. results may not be optimal. it is "
                        "recomended to use either a hisc as a test or one of "
                        f"the distances: {_VALID_DISTANCES} as a metric."
                    )
                    warnings.warn(msg, UserWarning)
                metric_func = gaussian
            else:
                if test != "hsic":
                    msg = (
                        f"{test} is a distance-baed test, but {metric} "
                        "is a kernel. results may not be optimal. it is "
                        "recomended to use either a hisc as a test or one of "
                        f"the distances: {_VALID_DISTANCES} as a metric."
                    )
                    warnings.warn(msg, UserWarning)

                def metric_func(X, Y=None, metric=metric, workers=None):
                    return pairwise_kernels(X, Y, metric=metric, n_jobs=workers)

        self.test = KSample(test, compute_distance=metric_func)
        self.n_bootstraps = n_bootstraps
        self.workers = workers
        self.order_correction = order_correction

    def _embed(self, A1, A2):
        # if not is_symmetric(A1) or not is_symmetric(A2):
        #     msg = "currently, testing is only supported for undirected graphs"
        #     raise NotImplementedError(msg)  # TODO asymmetric case

        if self.n_components is None:
            num_dims1 = select_dimension(A1)[0][-1]
            num_dims2 = select_dimension(A2)[0][-1]
            self.n_components = max(num_dims1, num_dims2)

        ase = AdjacencySpectralEmbed(n_components=self.n_components)
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

    def _sample_modified_ase(self, X, Y, pooled=False):
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
            # TODO unclear whether using pooled estimator provides more power.
            # TODO this should be investigated. should not matter under null.
            two_samples = np.concatenate([X, Y], axis=0)
            get_sigma = _fit_plug_in_variance_estimator(two_samples)
        else:
            get_sigma = _fit_plug_in_variance_estimator(X)
        X_sigmas = get_sigma(X) * (N - M) / (N * M)

        # increase the variance of X by sampling from the asy dist
        X_sampled = np.zeros(X.shape)
        # TODO may be parallelized, but requires keeping track of random state
        for i in range(N):
            X_sampled[i, :] = X[i, :] + stats.multivariate_normal.rvs(cov=X_sigmas[i])

        # return the embeddings in the appropriate order
        return (Y, X_sampled) if reverse_order else (X_sampled, Y)

    def fit(self, A1, A2):
        """
        Fits the test to the two input graphs

        Parameters
        ----------
        A1, A2 : nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            The two graphs to run a hypothesis test on.

        Returns
        -------
        self
        """
        A1 = import_graph(A1)
        A2 = import_graph(A2)

        X1_hat, X2_hat = self._embed(A1, A2)
        X1_hat, X2_hat = _median_sign_flips(X1_hat, X2_hat)

        if self.order_correction:
            X1_hat, X2_hat = self._sample_modified_ase(X1_hat, X2_hat)

        data = self.test.test(
            X1_hat, X2_hat, reps=self.n_bootstraps, workers=self.workers, auto=False
        )

        self.null_distribution_ = self.test.indep_test.null_dist
        self.sample_T_statistic_ = data[0]
        self.p_value_ = data[1]

        return self


def _median_sign_flips(X1, X2):
    X1_medians = np.median(X1, axis=0)
    X2_medians = np.median(X2, axis=0)
    val = np.multiply(X1_medians, X2_medians)
    t = (val > 0) * 2 - 1
    X1 = np.multiply(t.reshape(-1, 1).T, X1)
    return X1, X2


def _fit_plug_in_variance_estimator(X):
    """
    Takes in ASE of a graph and returns a function that estimates
    the variance-covariance matrix at a given point using the
    plug-in estimator from the RDPG Central Limit Theorem.

    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        adjacency spectral embedding of a graph

    Returns
    -------
    plug_in_variance_estimtor: functions
        a function that estimates variance (see below)
    """

    n = len(X)

    # precompute the Delta and the middle term matrix part
    delta = 1 / (n) * (X.T @ X)
    delta_inverse = np.linalg.inv(delta)
    middle_term_matrix = np.einsum("bi,bo->bio", X, X)

    def plug_in_variance_estimator(x):
        """
        Takes in a point of a matrix of points in R^d and returns an
        estimated covariance matrix for each of the points

        Parameters:
        -----------
        x: np.ndarray, shape (n, d)
            points to estimate variance at
            if 1-dimensional - reshaped to (1, d)

        Returns:
        -------
        covariances: np.ndarray, shape (n, d, d)
            n estimated variance-covariance matrices of the points provided
        """
        if x.ndim < 2:
            x = x.reshape(1, -1)
        # the following two lines are a properly vectorized version of
        # middle_term = 0
        # for i in range(n):
        #     middle_term += np.multiply.outer((x @ X[i] - (x @ X[i]) ** 2),
        #                                      np.outer(X[i], X[i]))
        # where the matrix part does not involve x and has been computed above
        middle_term_scalar = x @ X.T - (x @ X.T) ** 2
        middle_term = np.tensordot(middle_term_scalar, middle_term_matrix, axes=1)
        covariances = delta_inverse @ (middle_term / n) @ delta_inverse
        return covariances

    return plug_in_variance_estimator
