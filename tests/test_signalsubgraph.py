import unittest

import numpy as np

import graspy as gs
from graspy.classify.vertex_screen import VertexScreener
from graspy.simulations.simulations import sbm


class TestVertexScreener(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        # create a graph where the first 20
        # vertices are the signal  vertices
        K = 2
        sample_size = 200
        cls.n = [20, 80]
        cls.p = [0.3, 0.4]
        cls.P0 = np.array([cls.p[0], 0.2, 0.2, 0.3]).reshape(K, K)
        cls.P1 = np.array([cls.p[1], 0.2, 0.2, 0.3]).reshape(K, K)
        cls.Y0 = np.zeros(sample_size)
        cls.Y1 = np.ones(sample_size)
        cls.A0 = [sbm(cls.n, cls.P0) for i in range(sample_size)]
        cls.A1 = [sbm(cls.n, cls.P1) for i in range(sample_size)]
        cls.graphs = np.array(cls.A0 + cls.A1)
        cls.y = np.concatenate((cls.Y0, cls.Y1))[:, None]

    def test_vertexscreener_sbm(self):
        classifier = VertexScreener(num_vertices=20, distance_metric="dcorr")
        classifier.fit(self.graphs, self.y)

        true_verticesofinterest = np.arange(20)
        print(classifier.vertices_of_interest)
        print(true_verticesofinterest)
        self.assertTrue(
            checkEqual(classifier.vertices_of_interest, true_verticesofinterest)
        )


def checkEqual(L1, L2):
    return len(L1) == len(L2) and sorted(L1) == sorted(L2)
