# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest
import numpy
from graspologic.layouts.auto import _get_bounds


class TestAuto(unittest.TestCase):
    def test_get_bounds(self):
        y = numpy.array([(1, 2), (4, 5), (-1, -2), (10, -20)])
        minx, miny, maxx, maxy = _get_bounds(y)
        self.assertEqual(-1, minx)
        self.assertEqual(-20, miny)
        self.assertEqual(10, maxx)
        self.assertEqual(5, maxy)


if __name__ == "__main__":
    unittest.main()
