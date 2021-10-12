# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

from graspologic.datasets import *


class TestDatasets(unittest.TestCase):
    def test_drosphila_left(self):
        graph = load_drosophila_left()
        graph, labels = load_drosophila_left(return_labels=True)

    def test_drosphila_right(self):
        graph = load_drosophila_right()
        graph, labels = load_drosophila_right(return_labels=True)

    def test_load_mice(self):
        data = load_mice()
