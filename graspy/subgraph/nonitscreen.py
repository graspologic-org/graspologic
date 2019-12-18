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
    corr_threshold: float or int
        User's desired threshold to use for selecting nodes.

    See Also
    --------
    graspy.subgraph.itscreen

    References
    ----------
    .. [1] S. Wang, C. Chen, A. Badea, Priebe, C.E., Vogelstein, J.T.  "Signal 
        Subgraph Estimation Via Vertex Screening," arXiv: 1801.07683 [stat.ME], 2018
    """

    def __init__(self, opt, corr_threshold):
        super().__init__(opt=opt)

        if not isinstance(corr_threshold, numbers.Real):
            raise ValueError("corr_threshold must be float or int")

        self.corr_threshold = corr_threshold

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

    def _fit_transform(self, X, y):
        "Finds the signal subgraph from the correlation values"
        self.fit(X, y)

        # Find indicies of correlation values greater than c
        subgraph_verts = np.arange(self.n_vertices, dtype="int64")
        ind = self.corrs > self.corr_threshold

        # Make that into row vector
        ind = ind.reshape(1, len(ind))
        subgraph_verts = subgraph_verts[ind[0]]

        # Store the indicies of the subgraph for viewing
        self.subgraph_verts = subgraph_verts

        subgraph_n_vert = len(subgraph_verts)
        S_hat = np.zeros((self.n_graphs, subgraph_n_vert, subgraph_n_vert))
        for i in range(self.n_graphs):
            S_hat[i] = X[i][subgraph_verts][:, subgraph_verts]

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
