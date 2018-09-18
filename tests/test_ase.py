import unittest
import graphstats as gs
import numpy as np
import networkx as nx
from graphstats.embed.ase import ASEEmbed
from graphstats.simulations.simulations import er_nm

class TestASEEmbed(unittest.TestCase):
    
    def test_ase_er(self):
        k = 3
        embed = ASEEmbed(k=k)
        n = 10
        M = 20
        A = er_nm(n, M) + 5
        embed._reduce_dim(A)
        self.assertEqual(embed.lpm.X.shape, (n, k))
        self.assertTrue(embed.lpm.is_symmetric())

    
    