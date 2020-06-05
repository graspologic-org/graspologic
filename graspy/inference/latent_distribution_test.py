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

import numpy as np

from ..embed import select_dimension, AdjacencySpectralEmbed
from ..utils import import_graph, is_symmetric
from .base import BaseInference
from sklearn.metrics import pairwise_distances
from sklearn.metrics.pairwise import PAIRED_DISTANCES
from hyppo.ksample import KSample

_VALID_METRICS = list(PAIRED_DISTANCES.keys())
_VALID_METRICS.append("gaussian")  # we have a gaussian kernel implemented too
_VALID_TESTS = ["cca", "dcorr", "hhg", "rv", "hsic", "mgc"]


class LatentDistributionTest(BaseInference):
    """
    Two-sample hypothesis test for the problem of determining whether two random
    dot product graphs have the same distributions of latent positions.

    This test can operate on two graphs where there is no known matching between
    the vertices of the two graphs.
    Currently, testing is only supported for undirected graphs.

    Read more in the :ref:`tutorials <inference_tutorials>`

    Parameters
    ----------
    test : str
        Backend hypothesis test to use, one of ["cca", "dcorr", "hhg", "rv", "hsic", "mgc"].
        These tests are typically used for independence testing, but here they
        are used for a two-sample hypothesis test on the latent positions of
        two graphs. See :class:`hyppo.ksample.KSample` for more information.

    metric : str or function, (default="euclidean")
        Distance metric to use, either a callable or a valid string.
        The callable should behave similarly to :func:`sklearn.metrics.pairwise_distances`,
        if a string should be one of the keys in `sklearn.metrics.pairwise.PAIRED_DISTANCES`
        or "gaussian" which will use a Gaussian kernel on Euclidean distances
        with an adaptively selected bandwidth.

    n_components : int or None, optional (default=None)
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.
        See :func:`~graspy.embed.selectSVD` for more information.

    n_bootstraps : int (default=200)
        Number of bootstrap iterations for the backend hypothesis test. See
        :class:`hyppo.ksample.KSample` for more information.

    workers : int, optional (default=1)
        Number of workers to use. If more than 1, parallelizes the code.

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

    """

    def __init__(
        self,
        test="dcorr",
        metric="euclidean",
        n_components=None,
        n_bootstraps=200,
        workers=1,
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
        elif n_bootstraps < 1:
            msg = "{} is invalid number of bootstraps, must be greater than 1"
            raise ValueError(msg.format(n_bootstraps))

        if not isinstance(workers, int):
            msg = "workers must be an int, not {}".format(type(workers))
            raise TypeError(msg)
        elif workers <= 0:
            msg = "{} is invalid number of workers, must be greater than 0"
            raise ValueError(msg.format(workers))

        super().__init__(embedding="ase", n_components=n_components)

        if callable(metric):
            metric_func = metric
        else:
            if metric == "gaussian":
                metric_func = _medial_gaussian_kernel
            else:

                def metric_func(X, Y=None, metric=metric, workers=None):
                    return pairwise_distances(X, Y, metric=metric)

        self.test = KSample(test, compute_distance=metric_func)
        self.n_bootstraps = n_bootstraps
        self.workers = workers

    def _embed(self, A1, A2):
        if not is_symmetric(A1) or not is_symmetric(A2):
            msg = "currently, testing is only supported for undirected graphs"
            raise NotImplementedError(msg)  # TODO asymmetric case

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
            raise ValueError("Input graphs do not have same directedness")

        return X1_hat, X2_hat

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

        data = self.test.test(
            X1_hat, X2_hat, reps=self.n_bootstraps, workers=self.workers, auto=False
        )

        self.null_distribution_ = self.test.indep_test.null_dist
        self.sample_T_statistic_ = data[0]
        self.p_value_ = data[1]

        return self


def _medial_gaussian_kernel(X, Y=None, workers=None):
    """Baseline medial gaussian kernel similarity calculation
    Y is dummy to mimic sklearn pairwise_distances"""
    l1 = pairwise_distances(X, Y=Y, metric="cityblock")
    mask = np.ones(l1.shape, dtype=bool)
    np.fill_diagonal(mask, 0)
    gamma = 1.0 / (2 * (np.median(l1[mask]) ** 2))
    K = np.exp(-gamma * pairwise_distances(X, Y=Y, metric="sqeuclidean"))
    return 1 - K / np.max(K)


def _median_sign_flips(X1, X2):
    X1_medians = np.median(X1, axis=0)
    X2_medians = np.median(X2, axis=0)
    val = np.multiply(X1_medians, X2_medians)
    t = (val > 0) * 2 - 1
    X1 = np.multiply(t.reshape(-1, 1).T, X1)
    return X1, X2
