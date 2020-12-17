# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest
from graspologic.layouts.nooverlap._quad_node import move_point_on_line, _QuadNode
from graspologic.layouts.nooverlap._node import _Node


class TestNoOverlap(unittest.TestCase):
    def setUp(self):
        self.qn = _QuadNode(
            [_Node(99, 3, 7, 2, 0, "red"), _Node(100, 2, 9, 3, 0, "blue")], 5, 50
        )
        cells = {}
        self.x, self.y = (3, 3)
        self.num_x, self.num_y = 5, 5
        self.min_x, self.min_y = 0, 0
        self.max_size = 1.5

    def test_find_grid_cell_and_center(self):
        x, y = 2, 5
        (
            cell_x,
            cell_y,
            cell_center_x,
            cell_center_y,
        ) = self.qn.find_grid_cell_and_center(
            self.min_x, self.min_y, self.max_size, x, y
        )
        self.assertEqual(0, cell_x)
        self.assertEqual(1, cell_y)
        self.assertEqual(0.0, cell_center_x)
        self.assertEqual(3.0, cell_center_y)

    def test_move_point_on_line(self):
        print(move_point_on_line([1, 1], [3, 3], 2.0))

    def test_find_free_cell(self):
        cells = {}
        result = self.qn.find_free_cell(
            cells,
            self.x,
            self.y,
            self.num_x,
            self.num_y,
            self.min_x,
            self.min_y,
            self.max_size,
        )
        self.assertTupleEqual(result, (3, 3, 9, 9), "Failure")

    def test_find_free_cell_one_used(self):
        cells = {}
        cells[(3, 3)] = 999
        result = self.qn.find_free_cell(
            cells,
            self.x,
            self.y,
            self.num_x,
            self.num_y,
            self.min_x,
            self.min_y,
            self.max_size,
        )
        self.assertTupleEqual(result, (4, 3, 12, 9), "Failure")

    def test_find_free_cell_two_used(self):
        cells = {}
        cells[(3, 3)] = 999
        cells[(4, 3)] = 999
        result = self.qn.find_free_cell(
            cells,
            self.x,
            self.y,
            self.num_x,
            self.num_y,
            self.min_x,
            self.min_y,
            self.max_size,
        )
        self.assertTupleEqual(result, (4, 4, 12, 12), "Failure")

    def test_find_free_cell_three_used(self):
        cells = {}
        cells[(3, 3)] = 999
        cells[(4, 3)] = 999
        cells[(4, 4)] = 999
        result = self.qn.find_free_cell(
            cells,
            self.x,
            self.y,
            self.num_x,
            self.num_y,
            self.min_x,
            self.min_y,
            self.max_size,
        )
        self.assertTupleEqual(result, (3, 4, 9, 12), "Failure")

    def test_find_free_cell_four_used(self):
        cells = {}
        cells[(3, 3)] = 999
        cells[(4, 3)] = 999
        cells[(4, 4)] = 999
        cells[(3, 4)] = 999
        result = self.qn.find_free_cell(
            cells,
            self.x,
            self.y,
            self.num_x,
            self.num_y,
            self.min_x,
            self.min_y,
            self.max_size,
        )
        self.assertTupleEqual(result, (2, 4, 6, 12), "Failure")

    def test_find_free_cell_five_used(self):
        cells = {}
        cells[(3, 3)] = 999
        cells[(4, 3)] = 999
        cells[(4, 4)] = 999
        cells[(3, 4)] = 999
        cells[(2, 4)] = 999
        result = self.qn.find_free_cell(
            cells,
            self.x,
            self.y,
            self.num_x,
            self.num_y,
            self.min_x,
            self.min_y,
            self.max_size,
        )
        self.assertTupleEqual(result, (2, 3, 6, 9), "Failure")

    def test_find_free_cell_six_used(self):
        cells = {}
        cells[(3, 3)] = 999
        cells[(4, 3)] = 999
        cells[(4, 4)] = 999
        cells[(3, 4)] = 999
        cells[(2, 4)] = 999
        cells[(2, 3)] = 999
        result = self.qn.find_free_cell(
            cells,
            self.x,
            self.y,
            self.num_x,
            self.num_y,
            self.min_x,
            self.min_y,
            self.max_size,
        )
        self.assertTupleEqual(result, (2, 2, 6, 6), "Failure")

    def test_find_free_cell_seven_used(self):
        cells = {}
        cells[(3, 3)] = 999
        cells[(4, 3)] = 999
        cells[(4, 4)] = 999
        cells[(3, 4)] = 999
        cells[(2, 4)] = 999
        cells[(2, 3)] = 999
        cells[(2, 2)] = 999
        result = self.qn.find_free_cell(
            cells,
            self.x,
            self.y,
            self.num_x,
            self.num_y,
            self.min_x,
            self.min_y,
            self.max_size,
        )
        self.assertTupleEqual(result, (3, 2, 9, 6), "Failure")

    def test_find_free_cell_nine_used(self):
        cells = {}
        cells[(3, 3)] = 999
        cells[(4, 3)] = 999
        cells[(4, 4)] = 999
        cells[(3, 4)] = 999
        cells[(2, 4)] = 999
        cells[(2, 3)] = 999
        cells[(2, 2)] = 999
        cells[(3, 2)] = 999
        cells[(4, 2)] = 999
        result = self.qn.find_free_cell(
            cells,
            self.x,
            self.y,
            self.num_x,
            self.num_y,
            self.min_x,
            self.min_y,
            self.max_size,
        )
        self.assertTupleEqual(result, (1, 4, 3, 12), "Failure")

    #
    def test_find_free_cell_16_used(self):
        cells = {}
        cells[(3, 3)] = 999
        cells[(4, 3)] = 999
        cells[(4, 4)] = 999
        cells[(3, 4)] = 999
        cells[(2, 4)] = 999
        cells[(2, 3)] = 999
        cells[(2, 2)] = 999
        cells[(3, 2)] = 999
        cells[(4, 2)] = 999
        cells[(1, 4)] = 999
        cells[(1, 3)] = 999
        cells[(1, 2)] = 999
        cells[(1, 1)] = 999
        cells[(2, 1)] = 999
        cells[(3, 1)] = 999
        cells[(4, 1)] = 999
        result = self.qn.find_free_cell(
            cells,
            self.x,
            self.y,
            self.num_x,
            self.num_y,
            self.min_x,
            self.min_y,
            self.max_size,
        )
        self.assertTupleEqual(result, (0, 4, 0, 12), "Failure")

    ##
    def test_find_free_cell_20_used(self):
        cells = {}
        cells[(3, 3)] = 999
        cells[(4, 3)] = 999
        cells[(4, 4)] = 999
        cells[(3, 4)] = 999
        cells[(2, 4)] = 999
        cells[(2, 3)] = 999
        cells[(2, 2)] = 999
        cells[(3, 2)] = 999
        cells[(4, 2)] = 999
        cells[(1, 4)] = 999
        cells[(1, 3)] = 999
        cells[(1, 2)] = 999
        cells[(1, 1)] = 999
        cells[(2, 1)] = 999
        cells[(3, 1)] = 999
        cells[(4, 1)] = 999
        cells[(0, 4)] = 999
        cells[(0, 3)] = 999
        cells[(0, 2)] = 999
        cells[(0, 1)] = 999
        result = self.qn.find_free_cell(
            cells, 0, 2, self.num_x, self.num_y, self.min_x, self.min_y, self.max_size
        )
        self.assertTupleEqual(result, (0, 0, 0, 0), "Failure")
