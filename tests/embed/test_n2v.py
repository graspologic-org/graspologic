# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest
import networkx as nx
import numpy as np
from graspologic.embed import n2v


class TestN2V(unittest.TestCase):
    def test_node2vec_embed(self):
        g = nx.florentine_families_graph()

        for s, t in g.edges():
            g.add_edge(s, t, weight=1)

        embedding = n2v.node2vec_embed(g, random_seed=1)

        embedding2 = n2v.node2vec_embed(g, random_seed=1)

        np.testing.assert_array_equal(embedding[0], embedding2[0])


if __name__ == "__main__":
    unittest.main()
