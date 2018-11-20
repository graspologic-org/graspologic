import unittest
import graspy as gs
import numpy as np
import networkx as nx
from graspy.embed.embed import BaseEmbed
from graspy.simulations.simulations import er_nm


class TestBaseEmbed(unittest.TestCase):
    def test_baseembed_er(self):
        n_components = 4
        embed = BaseEmbed(n_components=n_components)
        n = 10
        M = 20
        A = er_nm(n, M) + 5
        embed._reduce_dim(A)
        self.assertEqual(embed.latent_left_.shape, (n, n_components))
        self.assertTrue(embed.latent_right_ is None)