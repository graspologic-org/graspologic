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
    TODO: Write documentation.
    """

    def __init__():
        pass

    def _test(self, i, j, graphs, y):
        """Calculate p-value for a specific edge."""
        samples = graphs[:, i, j]
        edge = [samples[y == label] for label in self.classes_]
        _, pvalue = KSample("MGC").test(*edge)

    def fit(self, graphs, y, n_jobs=-1):
        """
        TODO: Write documentation.
        """

        graphs, y = self._check_input_graphs(graphs, y)

        # Make an iterator over the edges of the graphs
        if self._are_directed(graphs):
            indices = zip(*np.indices(self.n_vertices_, self.n_vertices_))
        else:
            indices = zip(*np.triu_indices(self.n_vertices_, 1))

        # Calculate a p-value for each edge
        pvals = Parallel(n_jobs=n_jobs)(
            delayed(self._test)(i, j, graphs, y) for (i, j) in indices
        )

        # Construct dataframe of results
        columns = ["i", "j", "p-value"]
        pvals = pd.DataFrame(pvals, columns=columns)

        return pvals
