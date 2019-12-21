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

from abc import abstractmethod

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from mgcpy.independence_tests.mgc import MGC
from mgc.independence import Dcorr, RV, CCA


class BaseSubgraph(BaseEstimator):
    """
    A base class for estimating the signal subgraph.
    
    Parameters
    ----------
    opt : string
        Desired test statistic to use on data. If "mgc", 
        mulstiscale graph correlation will be used. Otherwise, must be
        either "dcorr", "rv", or "cca".
 
    See Also
    --------
    graspy.subgraph.nonitscreen, graspy.subgraph.itscreen
    """

    def __init__(self, opt):
        statistic_opts_ = ["mgc", "dcorr", "rv", "cca"]

        if opt not in statistic_opts_:
            raise ValueError('opt must be either "mgc", "dcorr", "rv", or "cca".')

        self.opt = opt

    def _screen(self, X, y):
        """
        Performs non-iterative screening on graphs to estimate signal subgraph.
    
        Parameters
        ----------
        X: np.ndarray, shape (n_graphs, n_vertices, n_vertices)
            Tensor of adjacency matrices.
        y: np.ndarray, shape (n_graphs, 1)
            Vector of ground truth labels.
        
        Attributes
        ----------
        n_graphs: int
            Number of graphs in X
        n_vertices: int
            Dimension of each graph in X
        
        Returns
        -------
        corrs: np.ndarray, shape (n_vertices, 1)
            Vector of correlation values for each node.
        
        References
        ----------
        .. [1] S. Wang, C. Chen, A. Badea, Priebe, C.E., Vogelstein, J.T.  "Signal 
        Subgraph Estimation Via Vertex Screening," arXiv: 1801.07683 [stat.ME], 2018
        """

        # Error Checking
        if type(X) is not np.ndarray:
            raise TypeError("X must be numpy.ndarray")
        if type(y) is not np.ndarray:
            raise TypeError("y must be numpy.ndarray")

        check_array(X, dtype=int, ensure_2d=False, allow_nd=True)
        check_array(y, dtype=int)

        # Finding dimensions
        self.n_graphs = X.shape[0]
        self.n_vertices = X.shape[-1]

        if len(X.shape) != 3:
            raise ValueError("X must be a tensor")
        if X.shape[1] != X.shape[2]:
            raise ValueError("Entries in X must be square matrices")

        if y.shape != (self.n_graphs, 1):
            raise ValueError("y must have shape (n_graphs, 1)")

        corrs = np.zeros((self.n_vertices, 1))
        for i in range(self.n_vertices):

            # Stacks the ith row of each matrix in tensor,
            # creates matrix with dimension n_graphs by n_vertices
            mat = X[:, i]

            # Statistical measurement chosen by the user
            if self.opt == "mgc":
                mgc = MGC()
                c_u, independence_test_metadata = mgc.test_statistic(mat, y)
                corrs[i][0] = c_u
            else:
                if self.opt == "dcorr":
                    test = Dcorr()
                elif self.opt == "rv":
                    test = RV()
                else:
                    test = CCA()
                c_u = test._statistic(mat, y)
                corrs[i][0] = c_u

        return corrs

    @abstractmethod
    def fit(self, X, y):
        """
        A method for signal subgraph estimation.
        
        Parameters
        ----------
        X: np.ndarray
        y : np.ndarray 
        
        Returns
        -------
        self : returns an instance of self.
        """

        return self
