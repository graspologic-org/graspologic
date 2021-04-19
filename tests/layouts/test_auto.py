# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest
import networkx as nx
import numpy

from graspologic.layouts.auto import _get_bounds, layout_umap


class TestAuto(unittest.TestCase):
    def test_get_bounds(self):
        y = numpy.array([(1, 2), (4, 5), (-1, -2), (10, -20)])
        minx, miny, maxx, maxy = _get_bounds(y)
        self.assertEqual(-1, minx)
        self.assertEqual(-20, miny)
        self.assertEqual(10, maxx)
        self.assertEqual(5, maxy)

    def test_layout_umap_string_node_ids(self):
        graph = nx.florentine_families_graph()

        for s, t in graph.edges():
            graph.add_edge(s, t, weight=1)

        _, node_positions = layout_umap(graph=graph)

        self.assertEqual(len(node_positions), len(graph.nodes()))

if __name__ == "__main__":
    unittest.main()
