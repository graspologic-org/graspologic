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

    n_bootstraps : int (default=200)
        Number of bootstrap iterations.

    bandwidth : float, optional (default=0.5)
        Bandwidth to use for gaussian kernel. If None,
        the median heuristic will be used.

    Attributes
    ----------
    null_distribution_ : np.ndarray
        The distribution of T statistics generated under the null.

    sample_T_statistic_ : float
        The observed difference between the embedded positions of the two input graphs
        after an alignment (the type of alignment depends on `test_case`)

    p_ : float
        The overall p value from the test.

    Examples
    --------
    >>> npt = LatentDistributionTest(n_components=2, which_test='mgc')
    >>> p = npt.fit(A1, A2)

    See also
    --------
    graspy.embed.AdjacencySpectralEmbed
    graspy.embed.OmnibusEmbed
    graspy.embed.selectSVD
    """
    # TODO: reference Varjavand paper when it is on arxiv

    def __init__(self, n_components=None, which_test="mgc", graph=False):
        if n_components is not None:
            if not isinstance(n_components, int):
                msg = "n_components must an int, not {}.".format(type(n_components))
                raise TypeError(msg)
        if type(which_test) is not str:
            msg = "which_test must be a string, not {}.".format(type(which_test))
            raise TypeError(msg)
        if which_test not in ["mgc", "dcorr"]:
            msg = "{} is not a valid test, must be mgc or dcorr.".format(which_test)
            raise ValueError(msg)
        super().__init__(embedding="ase", n_components=n_components)
        self.which_test = which_test
        self.graph = graph

    def _k_sample_transform(self, x, y):
        u = np.concatenate([x, y], axis=0)
        v = np.concatenate([np.repeat(1, x.shape[0]), np.repeat(2, y.shape[0])], axis=0)
        if len(u.shape) == 1:
            u = u[..., np.newaxis]
        if len(v.shape) == 1:
            v = v[..., np.newaxis]
        return u, v

    def _embed(self, A1, A2):
        ase = AdjacencySpectralEmbed(n_components=self.n_components)
        X1_hat = ase.fit_transform(A1)
        X2_hat = ase.fit_transform(A2)
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
        if self.graph:
            A1 = import_graph(A1)
            A2 = import_graph(A2)
            if not is_symmetric(A1) or not is_symmetric(A2):
                raise NotImplementedError()  # TODO asymmetric case
            if self.n_components is None:
                num_dims1 = select_dimension(A1.astype(float))[0][-1]
                num_dims2 = select_dimension(A2.astype(float))[0][-1]
                self.n_components = max(num_dims1, num_dims2)
            X1_hat, X2_hat = self._embed(A1, A2)
            X1_hat, X2_hat = self._k_sample_transform(X1_hat, X2_hat)
        else: #you already gave me stuff
            X1_hat = A1
            X2_hat = A2
        if self.which_test is "dcorr":
            test = DCorr('ubiased')
        elif self.which_test == "mgc":
            test = MGC()
        t, t_meta = test.test_statistic(X1_hat, X2_hat, is_fast=False)
        p, p_meta = test.p_value(X1_hat, X2_hat, is_fast=False)
        self.sample_T_statistic_ =  t
        self.null_distribution_ = list(p_meta)
        self.p_ = p
        return self.p_
