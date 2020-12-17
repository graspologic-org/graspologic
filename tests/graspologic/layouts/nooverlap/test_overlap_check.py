import unittest
from graspologic.layouts.nooverlap import _node
from graspologic.layouts.nooverlap._quad_node import _QuadNode, is_overlap

class TestOverlapCheck(unittest.TestCase):

	def setUp(self):
		self.qn = _QuadNode([_node(99, 3, 7, 2, 0, 'red'), _node(100, 2, 9, 3, 0, 'blue')], 5, 50)

	def test_grid_is_overlap(self):
		overlaps = is_overlap(0, 0, 5, 20, 20, 5)
		self.assertFalse(overlaps)
		overlaps = is_overlap(0, 0, 5, 10, 10, 5)
		self.assertFalse(overlaps)
		overlaps = is_overlap(0, 0, 5, 0, 10, 5)
		self.assertTrue(overlaps) #barely touches
		overlaps = is_overlap(0, 0, 5, 10, 0, 5)
		self.assertTrue(overlaps)
		overlaps = is_overlap(0, 0, 4.999, 10, 0, 5)
		self.assertFalse(overlaps)
		overlaps = is_overlap(2, 2, 1, 4, 4, 1)
		self.assertFalse(overlaps)



	def test_overlap_check_list(self):

		to_check = _node(0, 2, 2, 1, -1, 'blue')
		others = [_node(1, 5, 5, 1, -1, 'blue'), _node(2, 6, 6, 1, -1, 'blue'),_node(3, 7, 7, 1, -1, 'blue')]
		overlapping_node = self.qn.is_overlapping_any_node(to_check, to_check.x, to_check.y, others)
		self.assertIsNone(overlapping_node)
		others += [_node(4, 3, 3, 1, -1, 'red')]
		overlapping_node = self.qn.is_overlapping_any_node(to_check, to_check.x, to_check.y, others)
		self.assertIsNotNone(overlapping_node)
		self.assertEqual(overlapping_node.nid, 4)


	def test_is_overlapping_any_node_and_index(self):
		to_check = _node(0, 2, 2, 1, -1, 'blue')
		others = [_node(1, 5, 5, 1, -1, 'blue'), _node(2, 6, 6, 1, -1, 'blue'),_node(3, 7, 7, 1, -1, 'blue')]
		ov_idx, idx = 0,len(others)
		ov_idx, overlapping_node = self.qn.is_overlapping_any_node_and_index(to_check, to_check.x, to_check.y, others, ov_idx,idx)
		self.assertEquals(2, ov_idx)

		ov_idx, overlapping_node = self.qn.is_overlapping_any_node_and_index(to_check, to_check.x, to_check.y, others, 2, 3)
		self.assertEquals(2, ov_idx)


	def test_just_outside_box(self):
		self.assertTrue(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 6, 4.9, 1)) #down
		self.assertTrue(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 4, 4, 1)) #down left
		self.assertTrue(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 4, 7.1, 1)) #left
		self.assertTrue(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 11, 7.1, 1)) #right
		self.assertTrue(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 7.1, 11, 1)) #up

		##inside
		self.assertFalse(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 6, 6, 1)) #inside
		self.assertFalse(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 9, 9, 1)) #inside

		## way outside
		self.assertFalse(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 2.5, 7.1, 1)) #far left
		self.assertFalse(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 12.5, 7, 1)) #right
		self.assertFalse(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 7.1, 15, 1)) #far up
		self.assertFalse(self.qn.is_just_outside_box(5, 5, 10, 10, 1, 7.1, -3, 1)) #far down

		#### TESTING NEGATIVE
		self.assertTrue(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -6, -11, 1)) #down
		self.assertTrue(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -4, -4, 1)) #up right
		self.assertTrue(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -11, -7.1, 1)) #left
		self.assertTrue(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -4, -7.1, 1)) #right
		self.assertTrue(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -7.1, -4, 1)) #up

		##inside
		self.assertFalse(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -6, -6, 1)) #inside
		self.assertFalse(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -9, -9, 1)) #inside

		## way outside
		self.assertFalse(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -2.5, -7.1, 1)) #far right
		self.assertFalse(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -12.5, -7, 1)) #left
		self.assertFalse(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -7.1, -15, 1)) #far down
		self.assertFalse(self.qn.is_just_outside_box(-10, -10, -5, -5, 1, -7.1, -1, 1)) #far up

	def test_get_nodes_just_outside_box(self):
		others = [
			_node(0, 6, 4.9, 1, 7, 'blue'),
			_node(1, 4, 4, 1, -1, 'blue'),
			_node(2, 4, 7.1, 1, -1, 'blue'),
			_node(3, 11, 7.1, 1, -1, 'blue'),
			_node(4, 7.1, 11, 1, -1, 'blue'),
			_node(5, 6, 6, 1, -1, 'blue'),
			_node(6, 9, 9, 1, -1, 'blue'),
			_node(7, 2.5, 7.1, 1, -1, 'blue'),
			_node(8, 7.1, 15, 1, -1, 'blue'),
			_node(9, 7.1, -3, 1, -1, 'blue')
		]
		local_quad = _QuadNode([_node(99, 5, 5, 1, 0, 'red'), _node(100, 10, 10, 1, 0, 'blue')], 5, 50)
		just_outside = local_quad.get_nodes_near_lines(others)

		self.assertEqual(5, len(just_outside))


#if __name__ == '__main__':
#	unittest.main()
