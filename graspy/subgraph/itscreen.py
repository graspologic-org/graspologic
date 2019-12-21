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
    delta: float or int
        Desired quantile to be used for iterative screening. Must be between
        0 and 1.
    subgraph_n_verts: int
        Number of vertices for the produed subgraph. Must be between 0 and n_vertices.

    See Also
    --------
    graspy.subgraph.nonitscreen

    References
    ----------
    .. [1] S. Wang, C. Chen, A. Badea, Priebe, C.E., Vogelstein, J.T.  "Signal 
        Subgraph Estimation Via Vertex Screening," arXiv: 1801.07683 [stat.ME], 2018
    """

    def __init__(self, opt, delta, subgraph_n_verts):
        super().__init__(opt=opt)

        if not isinstance(delta, numbers.Real):
            raise ValueError("delta must be float or int")
        if delta < 0 or delta > 1:
            raise ValueError("delta must be between 0 and 1")

        self.delta = delta

        if not isinstance(subgraph_n_verts, int):
            raise ValueError("subgraph_n_verts must be an int")

        self.subgraph_n_verts = subgraph_n_verts

    def fit(self, X, y):
        """
        Performs iterative screening on graphs to estimate signal subgraph.
    
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

        self._screen(X, y)

        if self.subgraph_n_verts < 0 or self.subgraph_n_verts > self.n_vertices:
            raise ValueError("subgraph_n_verts must be between 0 and n_vertices")

        # Create empty array to store correlation values
        cors = np.zeros((self.n_vertices, 1))
        subgraph_inds = np.arange(self.n_vertices, dtype="int64")
        itr = 1

        tmpq = float("-inf")
        Atmp = X

        while ((1 - self.delta) ** itr) * self.n_vertices > self.subgraph_n_verts:

            # Creating new temporary set of subgraphs
            screen = NonItScreen(self.opt, tmpq)
            Atmp = screen.fit_transform(Atmp, y)
            screen = screen.fit(Atmp, y)

            # Correlation values
            tmpcors = screen.corrs

            # Specified quantile of correlation values
            tmpq = np.quantile(tmpcors, self.delta)

            # Add weights to the remaining correlation values
            cors[subgraph_inds] = tmpcors + itr

            # Iterate
            ind = tmpcors > tmpq
            ind = ind.reshape(1, len(ind))
            subgraph_inds = subgraph_inds[ind[0]]
            itr += 1

        self.corrs = cors
        return self

    def _fit_transform(self, X, y):
        "Finds the signal subgraph from the correlation values"
        self.fit(X, y)

        # Returns the indicies of the subgraph_n_verts largest values
        ind = self.corrs.reshape(1, self.n_vertices)
        val = self.subgraph_n_verts
        ind = ind[0].argsort()[-val:][::-1]
        ind = np.sort(ind)
        subgraph_inds = ind.reshape(1, len(ind))[0]
        self.subgraph_inds = subgraph_inds

        n = len(subgraph_inds)
        S_hat = np.zeros((self.n_graphs, n, n))
        for i in range(self.n_graphs):
            S_hat[i] = X[i][subgraph_inds][:, subgraph_inds]

        return S_hat

    def fit_transform(self, X, y):
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

        return self._fit_transform(X, y)
