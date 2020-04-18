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
import pandas as pd
from hyppo import KSample
from joblib import Parallel, delayed

from .base import BaseConnectomics


class EdgeTest(BaseConnectomics):
    """
    Edge significance test for an arbitrary number of input graphs with
    matched vertex sets.
    """

    def __init__():
        pass

    def _test(self, i, j, graphs, y):
        """Calculate p-value for a specific edge."""
        samples = graphs[:, i, j]
        edge = [samples[y == label] for label in self.classes_]
        _, pvalue = KSample("MGC").test(*edge)

    def fit(self, graphs, y, num_workers=-1):
        """
        Calculate the significance of edges between populations using MGC.

        Parameters
        ----------
        graphs : list of nx.Graph or ndarray, or ndarray
            If list of nx.Graph, each Graph must contain same number of nodes.
            If list of ndarray, each array must have shape (n_vertices, n_vertices).
            If ndarray, then array must have shape (n_graphs, n_vertices, n_vertices).
        y : array-like of shape (n_samples,)
            Label vector relative to graphs.
        num_workers : int, optional (default=1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.

        Returns
        -------
        pvals : pd.DataFrame
            Dataframe of the p-value for each edge.
        """

        graphs, y = self._check_input_graphs(graphs, y)

        # Make an iterator over the edges of the graphs
        if self._are_directed(graphs):
            indices = zip(*np.indices(self.n_vertices_, self.n_vertices_))
        else:
            indices = zip(*np.triu_indices(self.n_vertices_, 1))

        # Calculate a p-value for each edge
        pvals = Parallel(n_jobs=num_workers)(
            delayed(self._test)(i, j, graphs, y) for (i, j) in indices
        )

        # Construct dataframe of results
        columns = ["i", "j", "p-value"]
        pvals = pd.DataFrame(pvals, columns=columns)

        return pvals
