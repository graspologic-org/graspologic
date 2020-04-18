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
from sklearn.utils.validation import check_X_y

from ..utils import import_graph


class BaseConnectomics(BaseEstimator):
    """
    Base class for connectomics tasks such as edge, vertex, and block
    significance tests.
    """

    def _check_input_graphs(self, graphs, y):
        """
        Checks if all graphs in list have same shapes and have corresponding
        label.

        Raises a ValueError if there are more than one shape in the input list,
        or if the list is empty or has one element. Also raises a ValueError
        if there are an unequal number of graphs and labels.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).
        y : array-like of shape (n_samples,)
            Label vector relative to graphs.

        Returns
        -------
        graphs : ndarray, shape (n_graphs, n_vertices, n_vertices)
        y : ndarray, shape (n_graphs,)

        Raises
        ------
        ValueError
            If all graphs do not have same shape, or input list is empty or has
            one element.
        """
        # Convert input to np.arrays
        # This check is needed because np.stack will always duplicate array in memory.
        if isinstance(graphs, (list, tuple)):
            if len(graphs) <= 1:
                msg = "Input {} must have at least 2 graphs, not {}.".format(
                    type(graphs), len(graphs)
                )
                raise ValueError(msg)
            graphs = [import_graph(g, copy=False) for g in graphs]
        elif isinstance(graphs, np.ndarray):
            if graphs.ndim != 3:
                msg = f"Input tensor must be 3-dimensional, not {graphs.ndim}-dimensional."
                raise ValueError(msg)
            elif graphs.shape[0] <= 1:
                msg = f"Input tensor must have at least 2 elements, not {graphs.shape[0]}."
                raise ValueError(msg)
            graphs = import_graph(graphs, copy=False)
        else:
            msg = f"Input must be a list or ndarray, not {type(graphs)}."
            raise TypeError(msg)

        # Jointly validate the graphs and labels
        graphs, y = check_X_y(graphs, y, ensure_2d=False, allow_nd=True)

        # Save attributes
        self.n_graphs_ = len(graphs)
        self.n_vertices_ = graphs[0].shape[0]
        self.classes_ = np.unique(y)

        return graphs, y

    @abstractmethod
    def fit(self, graphs, y):
        """
        Fits the model and returns p-values for the selected scale.
        """
