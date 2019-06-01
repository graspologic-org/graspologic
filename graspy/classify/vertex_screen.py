from .base import BaseSignalSubgraph
from ..utils import import_graph, is_symmetric
from mgcpy.independence_tests.dcorr import DCorr
from mgcpy.independence_tests.mgc import MGC
import numpy as np


class VertexScreener(BaseSignalSubgraph):
    def __init__(self, num_vertices=20, distance_metric="mgc"):
        self.distance_metric = distance_metric
        self.num_vertices = num_vertices

    def fit(self, graphs, y):
        self._vertices, self._ts = self._screen_vertices(graphs, y)
        num_vertices_ = np.array([len(i) for i in self._vertices], dtype="int")
        self.vertices_of_interest_ = self._vertices[
            np.nonzero(num_vertices_ == self.num_vertices)[0][0]
        ]
        return self

    def _get_distance_correlations(self, A, Y):
        # A is of shape (num_samples,n_vertices,n_vertices)
        if self.distance_metric == "mgc":
            # use mgc test statistic if chosen
            corr = MGC().test_statistic
        else:
            # else use DCorr
            corr = DCorr().test_statistic
        c = []
        # for each vertex get the correlations
        for i in range(A.shape[1]):
            cu = corr(A[:, i, :], Y)
            c.append(cu[0])
        return np.array(c)

    def _screen_vertices(self, A, Y, delta=0.5):
        if self.distance_metric == "mgc":
            corr = MGC().test_statistic
        else:
            corr = DCorr().test_statistic
        corrs = []
        k = 0
        # create a list of the vertices
        V = [np.arange(len(A[-1]))]
        # while the vertex list is not empty
        while len(V[k]) > 1:
            # get the correlation between the current vertex
            # list and the labels
            c = self._get_distance_correlations(A[:, V[k][:, None], V[k]], Y)
            idx = np.argmin(c)
            V.append(V[k][np.arange(len(V[k])) != idx])
            # store the correlation for this iteration
            corrs.append(c)
            k += 1
        ts = []
        for i in range(len(V)):
            A_vk = np.reshape(A[:, V[i][:, None], V[i]], (A.shape[0], -1))
            ts.append(corr(A_vk, Y)[0])
        return V, ts

    def _fit_transform(self, graphs, y):
        self.fit(graphs, y)
        return graphs[:, self.vertices_of_interest_, self.vertices_of_interest_]
        # return self.vertices_of_interest_

    def fit_transform(self, graphs, y):
        return self._fit_transform(graphs, y)

    def score(self):
        pass
