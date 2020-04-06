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

from ..embed import select_dimension, AdjacencySpectralEmbed
from ..utils import import_graph, is_symmetric
from .base import BaseInference
from .dists import euclidean
from hyppo.ksample import KSample

class LatentDistributionTest(BaseInference):
    """
    Two-sample hypothesis test for the problem of determining whether two random
    dot product graphs have the same distributions of latent positions.

    This test can operate on two graphs where there is no known matching between
    the vertices of the two graphs, or even when the number of vertices is different.
    Currently, testing is only supported for undirected graphs.

    Read more in the :ref:`tutorials <inference_tutorials>`

    Parameters
    ----------
    test : str, one of ["CCA", "Dcorr", "HHG", "RV", "Hsic", "MGC"]
        Independence test to use.

    distance : function, (default=`function <euclidean>`)
        Distance function to use.

    n_components : int or None, optional (default=None)
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.

    n_bootstraps : int (default=200)
        Number of bootstrap iterations.

    num_workers : int, optional (default=1)
        Number of workers to use. If more than 1, parallelizes the code.

    Attributes
    ----------
    sample_T_statistic_ : float
        The observed difference between the embedded latent positions of the two
        input graphs.

    p_value_ : float
        The overall p value from the test.

    null_distribution_ : None or ndarray, shape (n_bootstraps, )
        The distribution of T statistics generated under the null.
        None if an approximation of the p-value is used.

    References
    ----------
    .. [1] Panda, S., Palaniappan, S., Xiong, J., Bridgeford, E., Mehta, R.,
        Shen, C., & Vogelstein, J. (2019).
        "hyppo: A Comprehensive Multivariate Hypothesis Testing Python Package."
        arXiv:1907.02088.
    """
    def __init__(self, test, distance=euclidean, n_components=None, n_bootstraps=200, num_workers=1):
        supported_tests = ["CCA", "Dcorr", "HHG", "RV", "Hsic", "MGC"]
        if not isinstance(test, str):
            msg = "test must be a str, not {}".format(type(test))
            raise TypeError(msg)
        elif test not in supported_tests:
            mgs = "test must be one of {}".format(supported_tests)
            raise ValueError(msg)

        if not callable(distance):
            msg = "distance must be a callable function, not {}".format(type(distance))
            raise TypeError(msg)

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

        if not isinstance(num_workers, int):
            msg = "num_workers must be an int, not {}".format(type(num_workers))
            raise TypeError(msg)
        elif num_workers <= 0:
            msg = "{} is invalid number of workers, must be greater than 0"
            raise ValueError(msg.format(num_workers))
        elif num_workers > 1:
            raise NotImplementedError() # TODO env error parallelizing

        super().__init__(embedding="ase", n_components=n_components)
        self.test = KSample(test, compute_distance=distance)
        self.n_bootstraps = n_bootstraps
        self.num_workers = num_workers

    def _embed(self, A1, A2):
        if not is_symmetric(A1) or not is_symmetric(A2):
            raise NotImplementedError()  # TODO asymmetric case
        if self.n_components is None:
            # get the last elbow from ZG for each and take the maximum
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

    def _perm_stat(x, y):
        permy = np.random.permutation(y)
        return self.test._statistic(x, permy)

    def _bootstrap(x, y, reps, workers=1):
        """
        Calculate the p-value via permutation test.
        """
        stat = self.test._statistic(x, y)

        null_dist = np.array(
            Parallel(n_jobs=workers)(
                [delayed(self._perm_stat)(x, y) for rep in range(reps)]
            )
        )
        p_value = (null_dist >= stat).sum() / reps
        if p_value == 0:
            p_value = 1 / reps

        self.null_distribution_ = null_dist
        return stat, p_value

    def fit(self, A1, A2, return_null_dist=False):
        """
        Fits the test to the two input graphs

        Parameters
        ----------
        A1, A2 : nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
            The two graphs to run a hypothesis test on.

        return_null_dist : bool (default=False)
            Flag whether to save the null distribution via permutation test.
            If False, uses a faster approximation of the p-value where possile,
            and saves ``self.null_distribution_`` as ``None``.
            MGC has no approximation and will always return a null distribution.

        Returns
        -------
        p_value : float
            The p value corresponding to the specified hypothesis test
        """
        if not isinstance(return_null_dist, bool):
            msg = "return_null_dist must be a bool, not {}".format(type(return_null_dist))
            raise TypeError(msg)
        A1 = import_graph(A1)
        A2 = import_graph(A2)
        X1_hat, X2_hat = self._embed(A1, A2)

        if return_null_dist:
            data = self._bootstrap(x, y, reps=self.n_bootstraps,
                                                      workers=self.num_workers)
        else:
            data = self.test.test(X1_hat, X2_hat, reps=self.n_bootstraps,
                                          workers=self.num_workers)
            self.null_distribution_ = None
        self.sample_T_statistic_ = data[0]
        self.p_value_ = data[1]

        return self.p_value_
