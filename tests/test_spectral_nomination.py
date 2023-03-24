# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import itertools
import unittest

import numpy as np

from graspologic.embed.ase import AdjacencySpectralEmbed
from graspologic.nominate import SpectralVertexNomination
from graspologic.simulations.simulations import sbm

# global constants for tests
n_verts = 50
p = np.array([[0.7, 0.25, 0.2], [0.25, 0.8, 0.3], [0.2, 0.3, 0.85]])
labels = np.array([0] * n_verts + [1] * n_verts + [2] * n_verts)
adj = np.array(sbm(3 * [n_verts], p), dtype=int)
embeder = AdjacencySpectralEmbed()
pre_embeded = embeder.fit_transform(adj)


class TestSpectralVertexNominatorOutputs(unittest.TestCase):
    def _nominate(self, X, seed, nominator=None, k=None):
        if nominator is None:
            nominator = SpectralVertexNomination(n_neighbors=k)
        nominator.fit(X)
        n_verts = X.shape[0]
        nom_list, dists = nominator.predict(seed)
        self.assertEqual(nom_list.shape, (n_verts, seed.shape[0]))
        self.assertEqual(dists.shape, (n_verts, seed.shape[0]))
        return nom_list

    def test_seed_inputs(self):
        with self.assertRaises(IndexError):
            self._nominate(adj, np.zeros((1, 50), dtype=int))
        with self.assertRaises(TypeError):
            self._nominate(adj, np.random.random((10, 2)))

    def test_X_inputs(self):
        with self.assertRaises(IndexError):
            self._nominate(np.zeros((5, 5, 5), dtype=int), np.zeros(3, dtype=int))
        with self.assertRaises(TypeError):
            self._nominate([[0] * 10] * 10, np.zeros(3, dtype=int))
        # embedding should have fewer cols than rows.
        svn = SpectralVertexNomination(input_graph=False)
        with self.assertRaises(IndexError):
            self._nominate(
                np.zeros((10, 20), dtype=int),
                np.zeros(3, dtype=int),
                nominator=svn,
            )
        # adj matrix should be square
        with self.assertRaises(IndexError):
            self._nominate(np.zeros((3, 4), dtype=int), np.zeros(3, dtype=int))

    def _test_k(self):
        # k should be > 0
        with self.assertRaises(ValueError):
            self._nominate(adj, np.zeros(3, dtype=int), k=0)
        # k of wrong type
        with self.assertRaises(TypeError):
            self._nominate(adj, np.zeros(3, dtype=int), k="hello world")

    def test_constructor_inputs(self):
        with self.assertRaises(ValueError):
            svn = SpectralVertexNomination(embedder="hi")
            self._nominate(adj, np.zeros(3, dtype=int), nominator=svn)

    def test_constructor_inputs1(self):
        # embedder must be BaseSpectralEmbed or str
        with self.assertRaises(TypeError):
            svn = SpectralVertexNomination(embedder=45)

    def test_constructor_inputs2(self):
        # input graph param has wrong type
        with self.assertRaises(TypeError):
            svn = SpectralVertexNomination(input_graph=4)

    def test_basic_unattributed(self):
        """
        Runs two attributed seeds and two unattributed seeds with each nominator.
        Ensures all options work. Should be fast. Nested parametrization tests all
        combinations of listed parameters.
        """
        nominators = [
            SpectralVertexNomination(embedder="ASE"),
            SpectralVertexNomination(embedder="LSE"),
            SpectralVertexNomination(embedder=embeder),
        ]
        seeds = [
            np.array([8]),
            np.array([2, 6, 9, 15, 25]),
            np.arange(n_verts - 1, dtype=int),
        ]
        for nominator, seed in itertools.product(nominators, seeds):
            self._nominate(adj, seed, nominator)

    def test_pre_embedded(self):
        seeds = [
            np.array([8]),
            np.array([2, 6, 9, 15, 25]),
            np.arange(n_verts - 1, dtype=int),
        ]
        for seed in seeds:
            svn = SpectralVertexNomination(input_graph=False)
            self._nominate(pre_embeded, seed, nominator=svn)
