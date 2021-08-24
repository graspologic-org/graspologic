# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import unittest

import numpy as np
from beartype.roar import BeartypeCallHintPepParamException

from graspologic.pipeline.embed import Embeddings


class TestEmbeddings(unittest.TestCase):
    def setUp(self) -> None:
        self.fake_embeddings = np.array([[0, 1, 2, 3], [5, 4, 3, 2], [3, 5, 1, 2]])
        self.labels = np.array(["dax", "nick", "ben"])
        self.embeddings = Embeddings(self.labels, self.fake_embeddings)

    def test_embeddings_index(self):
        for i in range(0, 3):
            entry = self.embeddings[i]
            self.assertEqual(self.labels[i], entry[0])
            np.testing.assert_array_equal(self.fake_embeddings[i], entry[1])

    def test_embeddings_iterable(self):
        labels = []
        embeddings = []
        for label, embedding in self.embeddings:
            labels.append(label)
            embeddings.append(embedding)

        np.testing.assert_array_equal(self.labels, labels)
        np.testing.assert_array_equal(self.fake_embeddings, embeddings)

    def test_embeddings_size(self):
        self.assertEqual(3, len(self.embeddings))

    def test_view(self):
        expected = {
            "ben": np.array([3, 5, 1, 2]),
            "dax": np.array([0, 1, 2, 3]),
            "nick": np.array([5, 4, 3, 2]),
        }
        view = self.embeddings.as_dict()
        self.assertSetEqual(set(view.keys()), set(expected.keys()))
        for key in expected:
            np.testing.assert_array_equal(expected[key], view[key])

    def test_argument_types(self):
        with self.assertRaises(BeartypeCallHintPepParamException):
            Embeddings(None, None)
        with self.assertRaises(BeartypeCallHintPepParamException):
            Embeddings(np.array(["hello"]), None)
        with self.assertRaises(BeartypeCallHintPepParamException):
            Embeddings(["hello"], [1.0])
        with self.assertRaises(ValueError):
            Embeddings(np.array(["hello"]), np.array([[1.1, 1.2], [2.1, 2.2]]))
