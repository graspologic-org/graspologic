# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import unittest

import numpy as np

import graspy as gs
from graspy.embed.base import BaseEmbed
from graspy.simulations.simulations import er_nm, er_np


class TestBaseEmbed(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        # simple ERxN graph
        cls.n = 20
        cls.p = 0.5
        cls.A = er_np(cls.n, cls.p, directed=True, loops=False)

    def test_baseembed_er(self):
        n_components = 4
        embed = BaseEmbed(n_components=n_components)
        n = 10
        M = 20
        A = er_nm(n, M) + 5
        embed._reduce_dim(A)
        self.assertEqual(embed.latent_left_.shape, (n, n_components))
        self.assertTrue(embed.latent_right_ is None)

    def test_baseembed_er_directed(self):
        n_components = 4
        embed = BaseEmbed(n_components=n_components)
        n = 10
        M = 20
        A = er_nm(n, M, directed=True)
        embed._reduce_dim(A)
        self.assertEqual(embed.latent_left_.shape, (n, n_components))
        self.assertEqual(embed.latent_right_.shape, (n, n_components))
        self.assertTrue(embed.latent_right_ is not None)

    def test_baseembed(self):
        embed = BaseEmbed(n_components=None)
        n = 10
        M = 20
        A = er_nm(n, M) + 5
        embed._reduce_dim(A)

    def test_algorithms(self):
        embed = BaseEmbed(n_components=self.n, algorithm="full")
        embed._reduce_dim(self.A)
        self.assertEqual(embed.latent_left_.shape, (self.n, self.n))
        self.assertEqual(embed.latent_right_.shape, (self.n, self.n))

        # When algoritm != 'full', cannot decompose to all dimensions
        embed = BaseEmbed(n_components=self.n, algorithm="truncated")
        with self.assertRaises(ValueError):
            embed._reduce_dim(self.A)

        embed = BaseEmbed(n_components=self.n, algorithm="randomized")
        with self.assertRaises(ValueError):
            embed._reduce_dim(self.A)
