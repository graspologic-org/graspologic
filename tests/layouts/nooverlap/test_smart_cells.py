# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest
from graspologic.layouts.nooverlap._smart_quad_node import _SmartCell, Extent
from graspologic.layouts.nooverlap._node import _Node


class TestSmartCellCount(unittest.TestCase):
    #def setUp(self):
    #    self.smart_cell_0_16_10 = _SmartCell(Extent(0, 0, 16, 16), 10)

    def test_is_full(self):
        smart_cell = _SmartCell(Extent(0, 0, 16, 16), 10)
        self.assertFalse(smart_cell.is_full(), "Should be non-full")
        smart_cell.full = True
        self.assertTrue(smart_cell.is_full(), "Should be full")
        with self.assertRaises(Exception):
            smart_cell.add_node(_Node("id1", 4, 4, 4))

    def test_others_full(self):
        smart_cell = _SmartCell(Extent(0, 0, 16, 16), 16)
        self.assertFalse(smart_cell.are_others_full(0, 0), "Should be room" )
        nodes = [_Node(str(x), 2, 2, 2) for x in range(16)]
        for n in nodes:
            smart_cell.add_node(n)
        self.assertTrue(smart_cell.is_full(), "Should be full")

    def test_add_large_node(self):
        smart_cell = _SmartCell(Extent(0, 0, 16, 16), 10)
        n = _Node("fake node", 6, 6, 6)
        smart_cell.add_node(n)
        self.assertTrue(smart_cell.is_full(), "Should be full")

    def test_small_node(self):
        smart_cell = _SmartCell(Extent(0, 0, 16, 16), 16)
        n = _Node("fake node", 2, 2, 4)
        smart_cell.add_node(n)
        self.assertFalse(smart_cell.is_full(), "Should NOT be full")
        self.assertEqual(2, smart_cell.rows)
        self.assertEqual(2, smart_cell.columns)
        self.assertEqual(1, len(smart_cell.children), "should have one child")
        child_cell = smart_cell.get_child_cell(0, 0)
        self.assertTrue(child_cell.is_full(), "child should be full")

    def test_two_small_nodes(self):
        smart_cell = _SmartCell(Extent(0, 0, 16, 16), 16)
        n = _Node("fake node", 2, 2, 4)
        n2 = _Node("fake node2", 2.5, 2, 4)
        smart_cell.add_node(n)
        self.assertFalse(smart_cell.is_full(), "Should NOT be full")
        self.assertEqual(1, len(smart_cell.children), "should have one child")
        child_cell = smart_cell.get_child_cell(0, 0)
        self.assertTrue(child_cell.is_full(), "child should be full")
        smart_cell.add_node(n2)
        self.assertFalse(smart_cell.is_full(), "Should NOT be full")

    def test_four_small_nodes(self):
        smart_cell = _SmartCell(Extent(0, 0, 16, 16), 16)
        n = _Node("fake node", 2, 2, 4)
        n2 = _Node("fake node2", 9, 2, 4)
        n3 = _Node("fake node3", 2.6, 2, 4)
        n4 = _Node("fake node4", 2.7, 2, 4)
        smart_cell.add_node(n)
        self.assertFalse(smart_cell.is_full(), "Should NOT be full")
        self.assertEqual(1, len(smart_cell.children), "should have one child")
        child_cell = smart_cell.get_child_cell(0, 0)
        self.assertTrue(child_cell.is_full(), "child should be full")
        smart_cell.add_node(n2)
        self.assertFalse(smart_cell.is_full(), "Should NOT be full")
        smart_cell.add_node(n3)
        self.assertFalse(smart_cell.is_full(), "Should NOT be full")
        smart_cell.add_node(n4)
        self.assertTrue(smart_cell.is_full(), "Should be full")
        with self.assertRaises(Exception):
            smart_cell.add_node(n)

    def test_smaller_nodes(self):
        smart_cell = _SmartCell(Extent(0, 0, 16, 16), 16)
        nodes = [_Node(str(x), 2, 2, 2) for x in range(16)]
        for n in nodes:
            smart_cell.add_node(n)
        self.assertTrue(smart_cell.is_full(), "Should be full")

    def test_different_sized_nodes(self):
        smart_cell = _SmartCell(Extent(0, 0, 16, 16), 16)
        nodes = [
            _Node(0, 2, 2, 4),  #SW
            _Node(1, 10, 10, 4),  #NE
            _Node(2, 10, 2, 4),   #SE
            _Node(3, 2, 14, 2),   #split into NW
            _Node(4, 2, 10, 2),   #split into NW
            _Node(5, 6, 10, 2),   #split into NW
            _Node(6, 6, 14, 2),   #split into NW
        ]
        for n in nodes[:-1]:
            smart_cell.add_node(n)
        #print (f"is full : {smart_cell.is_full()}")
        #print (f"num kids; {len(smart_cell.children)}")
        smart_cell.add_node(nodes[-1])
        #print (f"###is full : {smart_cell.is_full()}")

        #for c in range(smart_cell.columns):
        #    for r in range(smart_cell.rows):
        #        print (f"({c},{r})num kids; {len(smart_cell.get_child_cell(c,r).children)}, full: {smart_cell.get_child_cell(c,r).is_full()}")
        self.assertTrue(smart_cell.is_full(), "Should be full")

    def test_more_different_sized_nodes(self):
        smart_cell = _SmartCell(Extent(0, 0, 16, 16), 16)
        nodes = [
            _Node(0, 2, 2, 4),  #SW
            _Node(1, 10, 10, 4),  #NE
            _Node(2, 10, 2, 4),   #SE
            _Node(3, 2, 14, 2),   #split into NW
            _Node(4, 2, 10, 2),   #split into NW
            _Node(5, 6, 10, 2),   #split into NW
            #_Node(6, 6, 14, 2),   #split into NW
            _Node(6, 5, 13, 1),   #split into NW.SW
            _Node(6, 5, 15, 1),   #split into NW.NW
            _Node(6, 7, 15, 1),   #split into NW.NE
            _Node(6, 7, 13, 1),   #split into NW.SE
        ]
        for n in nodes[:-1]:
            smart_cell.add_node(n)
        #print (f"is full : {smart_cell.is_full()}")
        #print (f"num kids; {len(smart_cell.children)}")
        smart_cell.add_node(nodes[-1])
        #print (f"###is full : {smart_cell.is_full()}")

        #for c in range(smart_cell.columns):
        #    for r in range(smart_cell.rows):
        #        print (f"({c},{r})num kids; {len(smart_cell.get_child_cell(c,r).children)}, full: {smart_cell.get_child_cell(c,r).is_full()}")
        self.assertTrue(smart_cell.is_full(), "Should be full")

if __name__ == "__main__":
    unittest.main()
