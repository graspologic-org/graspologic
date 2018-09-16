import unittest
import graphstats as gs
import numpy as np
import networkx as nx
from graphstats.embed.embed import BaseEmbed
from graphstats.simulations.simulations import er_nm

class TestBaseEmbed(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # base embedding
        cls.embed = BaseEmbed()

    def test_baseembed_er(self):
        self.embed.fit(er_nm(10, 20))