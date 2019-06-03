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

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.validation import check_is_fitted


class BaseSignalSubgraph(ABC, BaseEstimator, ClassifierMixin):
    """
    Base Signal Subgraph class.
    """

    @abstractmethod
    def fit(self, graphs, y):
        """
        A method for computing the signal subgraph.

        Parameters
        ----------
        graphs: np.ndarray of adjacency matrices

        y : label for each graph in graphs

        Returns
        -------
        self : BaseSignalSubgraph object
            Contains the signal subgraph vertices.

        See Also
        --------
        
        """
        # import graph(s) to an adjacency matrix using import_graph function
        # here
        return self

    def _fit_transform(self, graph, y):
        "Fits the model and returns the signal subgraph"
        self.fit(graph, y)

    def fit_transform(self, graph, y):
        """
        Fit the model with graphs and apply the transformation. 

        n_dimension is either automatically determined or based on user input.

        Parameters
        ----------
        graph: np.ndarray or networkx.Graph

        y : Ignored

        Returns
        -------
        out : np.ndarray, shape (n_vertices, n_dimension) OR tuple (len 2)
            where both elements have shape (n_vertices, n_dimension)
            A single np.ndarray represents the latent position of an undirected
            graph, wheras a tuple represents the left and right latent positions 
            for a directed graph
        """
        return self._fit_transform(graph, y)
