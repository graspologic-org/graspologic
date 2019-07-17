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
import logging

from ..embed import AdjacencySpectralEmbed, select_dimension
from ..utils import import_graph, is_symmetric
from .base import BaseInference
from mgcpy.hypothesis_tests.transforms import k_sample_transform
from mgcpy.independence_tests.mgc import MGC
from mgcpy.independence_tests.dcorr import DCorr


class LatentDistributionTest(BaseInference):
    """
    Two-sample hypothesis test for the problem of determining whether two random
    dot product graphs have the same distributions of latent positions [2]_.

    This test can operate on two graphs where there is no known matching between
    the vertices of the two graphs, or even when the number of vertices is different.
    Currently, testing is only supported for undirected graphs.

    Parameters
    ----------
    n_components : int or None, optional (default=None)
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.

    n_bootstraps : int, optional (default=200)
        Number of bootstraps to perform when computing a p-value.

    method : string, {'dcorr' (default), 'mgc'}
        Which 2-sample test to use in order to detect differences between latent
        position distributions. `dcorr` (distance correlation) is much faster, `mgc`
        (multiscale graph correlation) is likely more powerful, but often slow.

    pass_graph : bool, optional (default True)
        If True, expects graphs as inputs. If False, expects latent positions as inputs.
        Graphs are n x n ndarrays or networkx graph objects.
        Latent positions are n x p ndarrays representing a set of points.

    Attributes
    ----------
    null_distribution_ : np.ndarray
        The distribution of T statistics generated under the null.

    sample_T_statistic_ : float
        The observed difference between the embedded positions of the two input graphs.

    p_ : float
        The p value from the test.

    Examples
    --------
    >>> lpt = LatentDistributionTest(n_components=2, method='mgc')
    >>> p = lpt.fit(A1, A2)

    See also
    --------
    graspy.embed.AdjacencySpectralEmbed
    graspy.embed.OmnibusEmbed
    graspy.embed.selectSVD

    References
    ----------
    .. [1] Tang, M., Athreya, A., Sussman, D. L., Lyzinski, V., & Priebe, C. E. (2017).
           A nonparametric two-sample hypothesis testing problem for random graphs. 
           Bernoulli, 23(3), 1599-1630.

    .. [2] Shen, C., Priebe, C. E., & Vogelstein, J. T. (2019). From distance
           correlation to multiscale graph correlation. Journal of the American
           Statistical Association, 1-22.

    .. [3] Shen, C., & Vogelstein, J. T. (2018). The exact equivalence of distance and
           kernel methods for hypothesis testing. arXiv preprint arXiv:1806.05514.
    """

    # TODO: reference Varjavand paper when it is on arxiv

    def __init__(
        self, n_components=None, n_bootstraps=200, method="dcorr", pass_graph=True
    ):
        if n_components is not None:
            if not isinstance(n_components, int):
                msg = "n_components must an int, not {}.".format(type(n_components))
                raise TypeError(msg)
        if type(n_bootstraps) is not int:
            msg = "n_bootstraps must be an int, not {}".format(type(n_bootstraps))
            raise TypeError(msg)
        if n_bootstraps <= 0:
            msg = "n_bootstraps must be > 0, not {}".format(n_bootstraps)
            raise ValueError(msg)
        if type(method) is not str:
            msg = "method must be a string, not {}.".format(type(method))
            raise TypeError(msg)
        if method not in ["mgc", "dcorr"]:
            msg = "{} is not a valid test, must be mgc or dcorr.".format(method)
            raise ValueError(msg)
        if type(pass_graph) is not bool:
            msg = "pass_graph must be a bool, not {}".format(type(pass_graph))
            raise TypeError(msg)
        super().__init__(
            embedding="ase", n_components=n_components, pass_graph=pass_graph
        )
        self.method = method
        self.n_bootstraps = n_bootstraps

    def _embed(self, A1, A2):
        if self.n_components is None:
            num_dims1 = select_dimension(A1.astype(float))[0][-1]
            num_dims2 = select_dimension(A2.astype(float))[0][-1]
            self.n_components = max(num_dims1, num_dims2)
        ase = AdjacencySpectralEmbed(n_components=self.n_components)
        # check symmetry
        if is_symmetric(A1) != is_symmetric(A2):
            msg = "graphs have unequal parity of symmetry"
            raise NotImplementedError(msg)  # TODO fix this in future? Many cases.
        if is_symmetric(A1):
            X1_hat = ase.fit_transform(A1)
            X2_hat = ase.fit_transform(A2)
        else:
            X1_hats = ase.fit_transform(A1)
            X2_hats = ase.fit_transform(A2)
            X1_hat = np.concatenate((X1_hats[0], X1_hats[1]), axis=1)
            X2_hat = np.concatenate((X2_hats[0], X2_hats[1]), axis=1)
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
        p_ : float
            The p value corresponding to the specified hypothesis test
        """
        if self.pass_graph:
            A1 = import_graph(A1)
            A2 = import_graph(A2)
            X1_hat, X2_hat = self._embed(A1, A2)
            X1_hat, X2_hat = k_sample_transform(X1_hat, X2_hat)
        else:  # you already have latent positions
            if type(A1) is not np.ndarray or type(A2) is not np.ndarray:
                raise TypeError(
                    "Your inputs should be np arrays, not {} and {}".format(
                        type(A1), type(A2)
                    )
                )
            X1_hat = A1
            X2_hat = A2
            X1_hat, X2_hat = k_sample_transform(X1_hat, X2_hat)
        if self.method == "dcorr":
            test = DCorr("unbiased")
        elif self.method == "mgc":
            test = MGC()
        p, p_meta = test.p_value(
            X1_hat, X2_hat, replication_factor=self.n_bootstraps, is_fast=False
        )
        self.sample_T_statistic_ = p_meta["test_statistic"]
        self.null_distribution_ = p_meta["null_distribution"]
        self.p_ = p
        return self.p_
