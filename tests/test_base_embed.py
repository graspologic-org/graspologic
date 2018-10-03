import unittest
import graspy as gs
import numpy as np
import networkx as nx
from graspy.embed.embed import BaseEmbed
from graspy.simulations.simulations import er_nm


class TestBaseEmbed(unittest.TestCase):
    def test_baseembed_er(self):
        k = 4
        embed = BaseEmbed(k=k)
        n = 10
        M = 20
        A = er_nm(n, M) + 5
        embed._reduce_dim(A)
        self.assertEqual(embed.lpm.X.shape, (n, k))
        self.assertTrue(embed.lpm.is_symmetric())