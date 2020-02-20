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
from sklearn.base import BaseEstimator
import numpy as np
import scipy


class BasePredictor(BaseEstimator):
    """
    Base class for prediction tasks such as link prediction.

    Parameters
    ----------
    n_components : None (default), or int
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.

    max_iter : int (default=5000)
        Maximum number of iterations for gradient descent.
    """

    def __init__(self, max_iter=5000):
        if not isinstance(max_iter, (int, np.integer)):
            raise TypeError("max_iter must be int or np.integer")
        if max_iter <= 0:
            raise ValueError(
                "Maximum number of iterations must be greater than 0"
                )
        self.max_iter = max_iter

    @abstractmethod
    def _optimize(self, A, mask, max_iter):
        """
        Solve the optimization problem.

        Returns
        -------
        X_hat : array-like, shape (n_vertices, n_components)
            Estimated latent positions of A
        """

    @abstractmethod
    def fit(self, A, mask):
        """
        Compute link predictions on a graph.

        Parameters
        ----------
        A : nx.Graph or np.ndarray
            A graph or adjacency matrix.
        mask : nx.Graph or np.ndarray
            A mask with 1 for known entries and 0 for unknown entries.
        """

    @abstractmethod
    def fit_transform(self, A, mask):
        """
        Compute link predictions on a graph and impute links.

        Parameters
        ----------
        A : nx.Graph or np.ndarray
            A graph or adjacency matrix.
        mask : nx.Graph or np.ndarray
            A mask with 1 for known entries and 0 for unknown entries.

        Returns
        -------
        A : np.ndarray
            An adjacency matrix of imputed links.
        """
