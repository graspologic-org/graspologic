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
from ..embed import OmnibusEmbed


class VertexTest(BaseConnectomics):
    """
    Vertex significance test for an arbitrary number of input graphs with
    matched vertex sets.
    """

    def __init__():
        pass

    def _test(self, vertex, embedding, y):
        """Calculate p-value for a specific vertex."""
        samples = embedding[:, vertex, :]
        vertex = [samples[y == label] for label in self.classes_]
        _, pvalue = KSample("MGC").test(*vertex)

    def fit(self, graphs, y, workers=-1):
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
        workers : int, optional (default=1)
            The number of cores to parallelize the p-value computation over.
            Supply -1 to use all cores available to the Process.

        Returns
        -------
        pvals : pd.DataFrame
            Dataframe of the p-value for each vertex.
        """

        graphs, y = self._check_input_graphs(graphs, y)

        # Embed the graphs
        omni = OmnibusEmbed()
        embedding = omni.fit_transform(graphs)

        # Calculate p-values for each vertex
        pvals = Parallel(n_jobs=workers)(
            delayed(self._test)(vertex, embedding, y) for vertex in range(self.n_vertices_)
        )

        # Construct dataframe of results
        columns = ["vertex", "p-value"]
        pvals = pd.DataFrame(pvals, columns=columns)

        return pvals
