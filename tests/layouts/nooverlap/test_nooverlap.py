# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest
from graspologic.layouts.nooverlap._quad_node import (
    move_point_on_line,
    _QuadNode,
    node_positions_overlap,
)
from graspologic.layouts.nooverlap._node import _Node
from graspologic.layouts.nooverlap.nooverlap import remove_overlaps
from graspologic.layouts import NodePosition


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

    def test_remove_overlap(self):
        list_of_nodes = [
            NodePosition(node_id=99, x=3, y=7, size=2, community=0),
            NodePosition(node_id=100, x=2, y=9, size=3, community=0),
            NodePosition(node_id=101, x=2, y=9, size=3, community=0),
        ]

        overlap_count = 0
        answer = remove_overlaps(list_of_nodes)
        for idx, n1 in enumerate(answer):
            print(idx, n1)
            for idx2, n2 in enumerate(answer[idx + 1 :]):
                print("****", idx2, "****", n2)
                if node_positions_overlap(n1, n2):
                    overlap_count += 1
                    print("overlap ****", idx2, "****", n2)
        self.assertEqual(0, overlap_count, "We found overlaps")


if __name__ == "__main__":
    unittest.main()
