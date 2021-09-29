# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

from graspologic.layouts.nooverlap._grid import _GridBuckets
from graspologic.layouts.nooverlap._node import _Node


class TestGrid(unittest.TestCase):

    # def setUp(self):
    # 	self.g = _GridBuckets(10)

    def test_get_cell(self):
        g = _GridBuckets(10)
        cell = g.get_cell(0, 0)
        self.assertTupleEqual((0, 0), cell)

        cell = g.get_cell(-1, -1)
        self.assertTupleEqual((-10, -10), cell)

        cell = g.get_cell(11, 11)
        self.assertTupleEqual((10, 10), cell)

        cell = g.get_cell(105, 87)
        self.assertTupleEqual((100, 80), cell)

        cell = g.get_cell(-105, -87)
        self.assertTupleEqual((-110, -90), cell)

        cell = g.get_cell(-105, 87)
        self.assertTupleEqual((-110, 80), cell)

        cell = g.get_cell(105, -57)
        self.assertTupleEqual((100, -60), cell)

    def test_get_grid_cells(self):
        g = _GridBuckets(10)
        cells = g._get_grid_cells(5, 12, 1)
        self.assertSetEqual({(0, 10)}, cells)

        g2 = _GridBuckets(20)
        cells = g2._get_grid_cells(5, 12, 10)
        self.assertSetEqual({(-20, 20), (0, 20), (0, 0), (-20, 0)}, cells)

        g3 = _GridBuckets(20)
        cells = g3._get_grid_cells(-5, -12, 10)
        self.assertSetEqual({(-20, -20), (0, -20), (0, -40), (-20, -40)}, cells)

    def test_add_node(self):
        g = _GridBuckets(10)
        n0 = _Node(0, 1, 1, 10, 1, "blue")
        n1 = _Node(1, 2, 1, 10, 1, "blue")
        n2 = _Node(2, 40, -20, 10, 1, "blue")

        g.add_node(n0)
        nodes = g.get_potential_overlapping_nodes_by_node(n0)
        self.assertSetEqual(nodes, {n0})

        g.add_node(n1)
        nodes = g.get_potential_overlapping_nodes_by_node(n1)
        self.assertSetEqual(nodes, {n0, n1})

        g.add_node(n2)
        nodes = g.get_potential_overlapping_nodes_by_node(n1)
        self.assertSetEqual(nodes, {n0, n1})

    def test_get_cell_stats(self):
        g = _GridBuckets(10)
        n0 = _Node(0, 1, 1, 10, 1, "blue")
        n1 = _Node(1, 2, 1, 10, 1, "blue")
        n2 = _Node(2, 40, -20, 10, 1, "blue")
        n3 = _Node(3, -33, -33, 1, 1, "blue")
        n4 = _Node(4, -193, 78, 1, 1, "blue")
        g.add_node(n0)
        g.add_node(n1)
        g.add_node(n2)
        g.add_node(n3)
        g.add_node(n4)
        stats = g.get_grid_cell_stats()
        self.assertEqual(3, len(stats), "Correct size list")
        self.assertEqual(254, stats[0][1], "empty cells")
        self.assertEqual(6, stats[1][1], "one item in cell")
        self.assertEqual(4, stats[2][1], "two items in cell")
        self.assertEqual(
            [(0, 254), (1, 6), (2, 4)], stats, "grid cell stats are in expected format"
        )


if __name__ == "__main__":
    unittest.main()
