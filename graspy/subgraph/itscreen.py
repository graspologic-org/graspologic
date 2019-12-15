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

from .base import BaseSubgraph
from . import NonItScreen


class ItScreen(BaseSubgraph):
    """
    Class to estimate signal subgraph for a covariate using iterative 
    vertex screening.
    
    Iterative screening uses the rows of the adjacency matricies as feature 
    vectors for each node. This algorithm then calculates the correlation between
    the features and the label vector. Any node whose correlation value is in a 
    designated quantile of the correlation values is kept and the non-iterative
    algorithm is run on the remaining nodes.
    Read more in the :ref:`tutorials <subgraph_tutorials>`
    
    Parameters
    ----------
    opt : string
        Desired test statistic to use on data. If "mgc", 
        mulstiscale graph correlation will be used. Otherwise, must be
        either "dcorr", "rv", or "cca".

    See Also
    --------
    graspy.subgraph.nonitscreen

    References
    ----------
    .. [1] S. Wang, C. Chen, A. Badea, Priebe, C.E., Vogelstein, J.T.  "Signal 
        Subgraph Estimation Via Vertex Screening," arXiv: 1801.07683 [stat.ME], 2018
    """

    def __init__(self, opt):
        super().__init__(opt=opt)

    def fit(self, X, y, delta, subgraph_n_verts):
        """
        Performs iterative screening on graphs to estimate signal subgraph.
    
        Parameters
        ----------
        X: np.ndarray, shape (n_graphs, n_vertices, n_vertices)
            Tensor of adjacency matrices
        y: np.ndarray, shape (n_graphs, 1)
            Vector of ground truth labels
        delta: int or float between 0 and 1
            Quantile to screen out the vertices each time
        subgraph_n_verts: int between 0 and n_vertices
            Size of signal subgraph that will be returned
    
        Returns
        -------
        self : returns an instance of self.
        """

        # Dimensions
        M = X.shape[0]
        N = X.shape[-1]

        # Create empty array to store correlation values
        cors = np.zeros((N, 1))
        subgraph_inds = np.arange(N, dtype="int64")
        itr = 1

        screen = NonItScreen(self.opt)
        tmpq = float("-inf")
        Atmp = X

        while ((1 - delta) ** itr) * N > subgraph_n_verts:

            # Creating new temporary set of subgraphs
            Atmp = screen.fit_transform(Atmp, y, tmpq)
            screen = screen.fit(Atmp, y)

            # Correlation values
            tmpcors = screen.corrs

            # Specified quantile of correlation values
            tmpq = np.quantile(tmpcors, delta)

            # Add weights to the remaining correlation values
            cors[subgraph_inds] = tmpcors + itr

            # Iterate
            ind = tmpcors > tmpq
            ind = ind.reshape(1, len(ind))
            subgraph_inds = subgraph_inds[ind[0]]
            itr += 1

        self.corrs = cors
        return self

    def _fit_transform(self, X, y, delta, subgraph_n_verts):
        "Finds the signal subgraph from the correlation values"
        self.fit(X, y, delta, subgraph_n_verts)

        # Dimensions
        M = X.shape[0]
        N = X.shape[-1]

        # Returns the indicies of the subgraph_n_verts largest values
        ind = self.corrs.reshape(1, N)
        ind = ind[0].argsort()[-20:][::-1]
        ind = np.sort(ind)
        subgraph_inds = ind.reshape(1, len(ind))[0]
        self.subgraph_inds = subgraph_inds

        n = len(subgraph_inds)
        S_hat = np.zeros((M, n, n))
        for i in range(M):
            S_hat[i] = X[i][subgraph_inds][:, subgraph_inds]

        return S_hat

    def fit_transform(self, X, y, delta, subgraph_n_verts):
        """
        Apply screening to graph set X to find nodes with 
        high enough correlation values.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_graphs, n_vertices, n_vertices)
        y : np.ndarray, shape (n_graphs, 1)
        delta : int or float between 0 and 1
        subgraph_n_verts : int between 0 and n_vertices
        
        Returns
        -------
        out : np.ndarray, shape (n_graphs, n_subgraph_inds, n_subgraph_inds)
        """

        return self._fit_transform(X, y, delta, subgraph_n_verts)
