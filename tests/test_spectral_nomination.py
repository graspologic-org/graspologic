import pytest
import numpy as np
from graspologic.embed.ase import AdjacencySpectralEmbed
from graspologic.simulations.simulations import sbm
from graspologic.nominate import SpectralVertexNominator


def _gen_att_seed(size, n_verts, labels):
    seed = np.empty((size, 2), dtype=np.int)
    seed[:, 0] = np.random.choice(3 * n_verts, size=size, replace=False).astype(np.int)
    seed[:, 1] = labels[seed[:, 0]]
    seed[-3:, 1] = np.array([0, 1, 2])  # ensure all atts represented
    return seed


class TestSpectralVertexNominator:
    # class vars
    n_verts = 50
    p = np.array([[0.7, 0.25, 0.2], [0.25, 0.8, 0.3], [0.2, 0.3, 0.85]])
    labels = np.array([0] * n_verts + [1] * n_verts + [2] * n_verts)
    adj = sbm(3 * [n_verts], p)
    embeder = AdjacencySpectralEmbed()
    pre_embeded = embeder.fit_transform(adj)

    @classmethod
    def _nominate(cls, seed, nominator):
        nominator.fit(cls.adj, seed)
        nom_list, dists = nominator.predict()
        unique_att = nominator.unique_att
        assert nom_list.shape == (3 * cls.n_verts, unique_att.shape[0])
        assert dists.shape == (3 * cls.n_verts, unique_att.shape[0])
        return nom_list

    @pytest.mark.parametrize(
        "nominator",
        [
            SpectralVertexNominator(embeder="ASE", persistent=False),
            SpectralVertexNominator(embeder="LSE", persistent=False),
            SpectralVertexNominator(embeder=embeder, persistent=False),
            SpectralVertexNominator(embedding=pre_embeded),
        ],
    )
    @pytest.mark.parametrize(
        "seed",
        [
            np.array([8]),
            np.array([2, 6, 9, 15, 25]),
            _gen_att_seed(10, n_verts, labels),
            _gen_att_seed(20, n_verts, labels),
        ],
    )
    def test_basic(self, nominator, seed):
        """
        Runs two attributed seeds and two unattributed seeds with each nominator.
        Ensures all options work. Should be fast. Nested parametrization tests all
        all combinations of listed parameters.
        """
        TestSpectralVertexNominator._nominate(seed, nominator)

    @pytest.mark.parametrize(
        "nominator",
        [
            SpectralVertexNominator(embeder="ASE", persistent=False),
            SpectralVertexNominator(embeder="LSE", persistent=False),
        ],
    )
    @pytest.mark.parametrize(
        "seed", [_gen_att_seed(20, n_verts, labels), _gen_att_seed(50, n_verts, labels)]
    )
    def test_attributed_acc(self, nominator, seed):
        """
        want to ensure performance is at least better than random
        """
        n_verts = TestSpectralVertexNominator.n_verts
        group1_correct = np.zeros(3 * n_verts)
        group2_correct = np.zeros(3 * n_verts)
        group3_correct = np.zeros(3 * n_verts)
        labels = np.array([0] * n_verts + [1] * n_verts + [2] * n_verts)
        for i in range(100):
            TestSpectralVertexNominator.adj = sbm(
                3 * [n_verts], TestSpectralVertexNominator.p
            )
            n_list = TestSpectralVertexNominator._nominate(
                seed=seed, nominator=nominator
            )
            group1_correct[np.argwhere(labels[n_list.T[0]] == 0)] += 1
            group2_correct[np.argwhere(labels[n_list.T[1]] == 1)] += 1
            group3_correct[np.argwhere(labels[n_list.T[2]] == 2)] += 1
        g1_correct_prob = group1_correct / 100
        g2_correct_prob = group2_correct / 100
        g3_correct_prob = group3_correct / 100
        assert np.mean(g1_correct_prob[: n_verts - 10]) > 0.8
        assert np.mean(g2_correct_prob[: n_verts - 10]) > 0.8
        assert np.mean(g3_correct_prob[: n_verts - 10]) > 0.8

    def test_seed_params(self):
        svn = SpectralVertexNominator(embedding=TestSpectralVertexNominator.pre_embeded)
        with pytest.raises(IndexError):
            TestSpectralVertexNominator._nominate(
                np.zeros((5, 5, 5), dtype=np.int), svn
            )
        with pytest.raises(IndexError):
            TestSpectralVertexNominator._nominate(np.zeros((1, 50), dtype=np.int), svn)
        with pytest.raises(TypeError):
            TestSpectralVertexNominator._nominate(np.random.random((10, 2)), svn)

    def test_graph_params(self):
        with pytest.raises(IndexError):
            SpectralVertexNominator(embedding=np.zeros((5, 5, 5), dtype=np.int))

        with pytest.raises(IndexError):
            svn = SpectralVertexNominator()
            shape = TestSpectralVertexNominator.adj.shape
            svn.fit(
                np.reshape(TestSpectralVertexNominator.adj, (1, shape[0], shape[1])),
                np.array([1], dtype=np.int),
            )

        with pytest.raises(IndexError):
            svn = SpectralVertexNominator()
            svn.fit(TestSpectralVertexNominator.adj[1:], np.array([1], dtype=np.int))

        with pytest.raises(TypeError):
            svn = SpectralVertexNominator()
            svn.fit(
                TestSpectralVertexNominator.adj.astype(np.object),
                np.array([1], dtype=np.int),
            )

    def test_predict_params(self):
        svn = SpectralVertexNominator()
        with pytest.raises(TypeError):
            svn.predict(k=5.3)
        with pytest.raises(ValueError):
            svn.predict(k=0)
