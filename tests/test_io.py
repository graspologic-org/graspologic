import sys


import pytest
import graspy as gs
import numpy as np
import networkx as nx


class TestImportGraph:
    @classmethod
    def setup_class(cls):
        # simple ERxN graph
        n = 15
        p = 0.5
        cls.A = np.zeros((n, n))
        nedge = int(round(n * n * p))
        np.put(
            cls.A,
            np.random.choice(np.arange(0, n * n), size=nedge, replace=False),
            np.random.normal(size=nedge),
        )

    def test_graphin(self):
        G = nx.from_numpy_array(self.A)
        np.testing.assert_array_equal(nx.to_numpy_array(G), gs.utils.import_graph(G))

    def test_npin(self):
        np.testing.assert_array_equal(self.A, gs.utils.import_graph(self.A))

    def test_wrongtypein(self):
        a = 5
        with pytest.raises(TypeError):
            gs.utils.import_graph(a)
        with pytest.raises(TypeError):
            gs.utils.import_graph(None)

    def test_nonsquare(self):
        non_square = np.hstack((self.A, self.A))
        with pytest.raises(ValueError):
            gs.utils.import_graph(non_square)


class TestImportEdgelist:
    @pytest.fixture(autouse=True)
    def setup_class(self, tmp_path):

        n = 10
        p = 0.5
        wt = np.random.exponential
        wtargs = dict(scale=4)

        np.random.seed(1)

        self.A = gs.simulations.er_np(n, p)
        self.B = gs.simulations.er_np(n, p, wt=wt, wtargs=wtargs)

        G_A = nx.from_numpy_array(self.A)
        G_B = nx.from_numpy_array(self.B)

        self.A_path = str(tmp_path / "A_unweighted.edgelist")
        self.B_path = str(tmp_path / "B.edgelist")
        self.root = str(tmp_path)

        nx.write_edgelist(G_A, self.A_path, data=False)
        nx.write_weighted_edgelist(G_B, self.B_path)

    def test_in(self):
        A_from_edgelist = gs.utils.import_edgelist(self.A_path)
        B_from_edgelist = gs.utils.import_edgelist(self.B_path)

        assert np.allclose(A_from_edgelist, self.A)
        assert np.allclose(B_from_edgelist, self.B)

    def test_multiple_in(self):
        graphs = gs.utils.import_edgelist(self.root)
        assert len(graphs) == 2
        assert np.allclose(graphs[0], self.A)
        assert np.allclose(graphs[1], self.B)

    def test_wrongtypein(self):
        path = 5
        with pytest.raises(TypeError):
            gs.utils.import_edgelist(path)
        with pytest.raises(TypeError):
            gs.utils.import_edgelist(None)

    def test_vertices(self):
        expected_vertices = np.arange(0, 10)

        _, A_vertices = gs.utils.import_edgelist(str(self.A_path), return_vertices=True)
        _, B_vertices = gs.utils.import_edgelist(str(self.B_path), return_vertices=True)

        assert np.allclose(expected_vertices, A_vertices)
        assert np.allclose(expected_vertices, B_vertices)

    def test_no_graphs_found(self):
        path = str(self.root + "invalid_edgelist.edgelist")
        with pytest.raises(ValueError):
            gs.utils.import_edgelist(path)
