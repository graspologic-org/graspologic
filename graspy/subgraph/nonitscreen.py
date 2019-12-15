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
import numbers
from sklearn.utils import check_array
from mgcpy.independence_tests.mgc import MGC
from mgc.independence import Dcorr, RV, CCA

from .base import BaseSubgraph


class NonItScreen(BaseSubgraph):
    """
    Class to estimate signal subgraph for a covariate using non-iterative 
    vertex screening.
    
    Non-iterative screening uses the rows of the adjacency matricies as feature 
    vectors for each node. This algorithm then calculates the correlation between
    the features and the label vector. Any node that has a correlation higher
    than the threshold is included as the signal subgraph.
    Read more in the :ref:`tutorials <subgraph_tutorials>`
    
    Parameters
    ----------
    opt : string
        Desired test statistic to use on data. If "mgc", 
        mulstiscale graph correlation will be used. Otherwise, must be
        either "dcorr", "rv", or "cca".

    See Also
    --------
    graspy.subgraph.itscreen

    References
    ----------
    .. [1] S. Wang, C. Chen, A. Badea, Priebe, C.E., Vogelstein, J.T.  "Signal 
        Subgraph Estimation Via Vertex Screening," arXiv: 1801.07683 [stat.ME], 2018
    """

    def __init__(self, opt):
        super().__init__(opt=opt)

    def fit(self, X, y):
        """
        Performs non-iterative screening on graphs to estimate signal subgraph.
    
        Parameters
        ----------
        X: np.ndarray, shape (n_graphs, n_vertices, n_vertices)
            Tensor of adjacency matrices
        y: np.ndarray, shape (n_graphs, 1)
            Vector of ground truth labels

        Returns
        -------
        self : returns an instance of self.
        """

        self.corrs = self._screen(X, y)
        return self

    def _fit_transform(self, X, y, corr_threshold):
        "Finds the signal subgraph from the correlation values"
        self.fit(X, y)

        if not isinstance(corr_threshold, numbers.Real):
            raise ValueError("Input must be float or int")

        # Dimensions
        M = X.shape[0]
        N = X.shape[-1]

        # Finds indicies of correlation values greater than c and makes that into column vector
        subgraph_inds = np.arange(N, dtype="int64")
        ind = self.corrs > corr_threshold

        # Reshaping the indices
        ind = ind.reshape(1, len(ind))
        subgraph_inds = subgraph_inds[ind[0]]
        self.subgraph_inds = subgraph_inds

        n = len(subgraph_inds)
        S_hat = np.zeros((M, n, n))
        for i in range(M):
            S_hat[i] = X[i][subgraph_inds][:, subgraph_inds]

        return S_hat

    def fit_transform(self, X, y, corr_threshold):
        """
        Apply screening to graph set X to find nodes with 
        high enough correlation values.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_graphs, n_vertices, n_vertices)
        y : np.ndarray, shape (n_graphs, 1)
        
        Returns
        -------
        out : np.ndarray, shape (n_graphs, n_subgraph_inds, n_subgraph_inds)
        """

        return self._fit_transform(X, y, corr_threshold)
