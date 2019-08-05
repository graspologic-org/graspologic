# Copyright 2019 NeuroData (http://neurodata.io)
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

from ..embed import AdjacencySpectralEmbed, select_dimension
from ..utils import import_graph, is_symmetric
from .base import BaseInference


class LatentDistributionTest(BaseInference):
    """
    Two-sample hypothesis test for the problem of determining whether two random 
    dot product graphs have the same distributions of latent positions [1]_.
    
    This test can operate on two graphs where there is no known matching between
    the vertices of the two graphs, or even when the number of vertices is different. 
    Currently, testing is only supported for undirected graphs.

    Read more in the :ref:`tutorials <inference_tutorials>`

    Parameters
    ----------
    n_components : int or None, optional (default=None)
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.

    n_bootstraps : int (default=200)
        Number of bootstrap iterations.

    bandwidth : float, optional (default=0.5)
        Bandwidth to use for gaussian kernel. If None,
        the median heuristic will be used.

    Attributes
    ----------
    sample_T_statistic_ : float
        The observed difference between the embedded latent positions of the two 
        input graphs.

    p_ : float
        The overall p value from the test.
    
    null_distribution_ : ndarray, shape (n_bootstraps, )
        The distribution of T statistics generated under the null.

    References
    ----------
    .. [1] Tang, M., Athreya, A., Sussman, D. L., Lyzinski, V., & Priebe, C. E. (2017). 
        "A nonparametric two-sample hypothesis testing problem for random graphs."
        Bernoulli, 23(3), 1599-1630.
    """

    def __init__(self, n_components=None, n_bootstraps=200, bandwidth=None):
        if n_components is not None:
            if not isinstance(n_components, int):
                msg = "n_components must an int, not {}.".format(type(n_components))
                raise TypeError(msg)

        if not isinstance(n_bootstraps, int):
            msg = "n_bootstraps must an int, not {}".format(type(n_bootstraps))
            raise TypeError(msg)
        elif n_bootstraps < 1:
            msg = "{} is invalid number of bootstraps, must be greater than 1"
            raise ValueError(msg.format(n_bootstraps))

        if bandwidth is not None and not isinstance(bandwidth, float):
            msg = "bandwidth must an int, not {}".format(type(bandwidth))
            raise TypeError(msg)

        super().__init__(embedding="ase", n_components=n_components)
        self.n_bootstraps = n_bootstraps
        self.bandwidth = bandwidth

    def _gaussian_covariance(self, X, Y):
        diffs = np.expand_dims(X, 1) - np.expand_dims(Y, 0)
        if self.bandwidth is None:
            self.bandwidth = 0.5
        return np.exp(-0.5 * np.sum(diffs ** 2, axis=2) / self.bandwidth ** 2)

    def _statistic(self, X, Y):
        N, _ = X.shape
        M, _ = Y.shape
        x_stat = np.sum(self._gaussian_covariance(X, X) - np.eye(N)) / (N * (N - 1))
        y_stat = np.sum(self._gaussian_covariance(Y, Y) - np.eye(M)) / (M * (M - 1))
        xy_stat = np.sum(self._gaussian_covariance(X, Y)) / (N * M)
        return x_stat - 2 * xy_stat + y_stat

    def _embed(self, A1, A2):
        ase = AdjacencySpectralEmbed(n_components=self.n_components)
        X1_hat = ase.fit_transform(A1)
        X2_hat = ase.fit_transform(A2)
        if isinstance(X1_hat, tuple) and isinstance(X2_hat, tuple):
            X1_hat = np.concatenate(X1_hat, axis=-1)
            X2_hat = np.concatenate(X2_hat, axis=-1)
        elif isinstance(X1_hat, tuple) ^ isinstance(X2_hat, tuple):
            raise ValueError("Input graphs do not have same directedness")
        return X1_hat, X2_hat

    def _median_heuristic(self, X1, X2):
        X1_medians = np.median(X1, axis=0)
        X2_medians = np.median(X2, axis=0)
        val = np.multiply(X1_medians, X2_medians)
        t = (val > 0) * 2 - 1
        X1 = np.multiply(t.reshape(-1, 1).T, X1)
        return X1, X2

    def _bootstrap(self, X, Y, M=200):
        N, _ = X.shape
        M2, _ = Y.shape
        Z = np.concatenate((X, Y))
        statistics = np.zeros(M)
        for i in range(M):
            bs_Z = Z[
                np.random.choice(np.arange(0, N + M2), size=int(N + M2), replace=False)
            ]
            bs_X2 = bs_Z[:N, :]
            bs_Y2 = bs_Z[N:, :]
            statistics[i] = self._statistic(bs_X2, bs_Y2)
        return statistics

    def fit(self, A1, A2):
        """
        Fits the test to the two input graphs

        Parameters
        ----------
        A1, A2 : nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            The two graphs to run a hypothesis test on.

        Returns
        -------
        p_ : float
            The p value corresponding to the specified hypothesis test
        """
        A1 = import_graph(A1)
        A2 = import_graph(A2)
        # if not is_symmetric(A1) or not is_symmetric(A2):
        #     raise NotImplementedError()  # TODO asymmetric case
        if self.n_components is None:
            # get the last elbow from ZG for each and take the maximum
            num_dims1 = select_dimension(A1)[0][-1]
            num_dims2 = select_dimension(A2)[0][-1]
            self.n_components = max(num_dims1, num_dims2)

        X1_hat, X2_hat = self._embed(A1, A2)
        X1_hat, X2_hat = self._median_heuristic(X1_hat, X2_hat)
        U = self._statistic(X1_hat, X2_hat)
        null_distribution = self._bootstrap(X1_hat, X2_hat, self.n_bootstraps)
        self.null_distribution_ = null_distribution
        self.sample_T_statistic_ = U
        p_value = (len(null_distribution[null_distribution >= U])) / self.n_bootstraps
        if p_value == 0:
            p_value = 1 / self.n_bootstraps
        self.p_ = p_value
        return self.p_
