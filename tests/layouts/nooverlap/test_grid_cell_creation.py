# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

from graspologic.layouts.nooverlap._node import _Node
from graspologic.layouts.nooverlap._quad_node import _QuadNode


class TestGridCellCreation(unittest.TestCase):
    def setUp(self):
        self.qn = _QuadNode(
            [_Node(99, 3, 7, 2, 0, "red"), _Node(100, 2, 9, 3, 0, "blue")], 5, 50
        )

    def test_grid_cell_center(self):
        cell_x, cell_y, center_x, center_y = self.qn.find_grid_cell_and_center(
            0, 0, 10, 50, 50
        )
        self.assertEqual(cell_x, 2)
        self.assertEqual(cell_y, 2)
        self.assertEqual(center_x, 40)
        self.assertEqual(center_y, 40)

    def test_grid_cell_center2(self):
        cell_x, cell_y, center_x, center_y = self.qn.find_grid_cell_and_center(
            0, 0, 10, 50, 40
        )
        self.assertEqual(cell_x, 2)
        self.assertEqual(cell_y, 2)
        self.assertEqual(center_x, 40)
        self.assertEqual(center_y, 40)

    def test_grid_cell_center3(self):
        cell_x, cell_y, center_x, center_y = self.qn.find_grid_cell_and_center(
            3, 4, 10, 53, 44
        )
        self.assertEqual(cell_x, 2)
        self.assertEqual(cell_y, 2)
        self.assertEqual(center_x, 43)
        self.assertEqual(center_y, 44)


if __name__ == "__main__":
    unittest.main()
