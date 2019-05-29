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


class BaseInference(BaseEstimator):
    """
    Base class for inference tasks such as semiparametric and
    nonparametric inference tasks.

    Parameters
    ----------
    embedding : { 'ase' (default), 'lse, 'omnibus'}
        String describing the embedding method to use.
        Must be one of:
        'ase'
            Embed each graph separately using adjacency spectral embedding
            and use Procrustes to align the embeddings.
        'omnibus'
            Embed all graphs simultaneously using omnibus embedding.

    n_components : None (default), or int
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.
    """

    def __init__(self, embedding="ase", n_components=None):
        if type(embedding) is not str:
            raise TypeError("embedding must be str")
        if (not isinstance(n_components, (int, np.integer))) and (
            n_components is not None
        ):
            raise TypeError("n_components must be int or np.integer")
        if embedding not in ["ase", "omnibus"]:
            raise ValueError("{} is not a valid embedding method.".format(embedding))
        if n_components is not None and n_components <= 0:
            raise ValueError(
                "Cannot embed into {} dimensions, must be greater than 0".format(
                    n_components
                )
            )
        self.embedding = embedding
        self.n_components = n_components

    @abstractmethod
    def _bootstrap(self):
        pass

    @abstractmethod
    def _embed(self, X1, X2, n_componets):
        """
        Computes the latent positions of input graphs

        Returns
        -------
        X1_hat : array-like, shape (n_vertices, n_components)
            Estimated latent positions of X1
        X2_hat : array-like, shape(n_vertices, n_components)
            Estimated latent positions of X2
        """

    @abstractmethod
    def fit(self, A1, A2):
        """
        Compute the test statistic and the null distribution.

        Parameters
        ----------
        X1 : nx.Graph or np.ndarray
            A graph
        X2 : nx.Graph or np.ndarray
            A graph
        """
