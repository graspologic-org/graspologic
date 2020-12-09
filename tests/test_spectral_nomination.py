import pytest
import numpy as np
from graspologic.embed.ase import AdjacencySpectralEmbed
from graspologic.simulations.simulations import sbm
from graspologic.nominate import SpectralVertexNomination

# global constants for tests
n_verts = 50
p = np.array([[0.7, 0.25, 0.2], [0.25, 0.8, 0.3], [0.2, 0.3, 0.85]])
labels = np.array([0] * n_verts + [1] * n_verts + [2] * n_verts)
adj = np.array(sbm(3 * [n_verts], p), dtype=np.int)
embeder = AdjacencySpectralEmbed()
pre_embeded = embeder.fit_transform(adj)


def _gen_att_seed(size, n_verts, labels):
    seed = np.empty((size, 2), dtype=np.int)
    seed[:, 0] = np.random.choice(3 * n_verts, size=size, replace=False).astype(np.int)
    seed[:, 1] = labels[seed[:, 0]]
    seed[-3:, 1] = np.array([0, 1, 2])  # ensure all atts represented
    return seed


def _nominate(X, seed, nominator=None, k=None):
    if nominator is None:
        nominator = SpectralVertexNomination()
    nominator.fit(X, seed, k=k)
    n_verts = X.shape[0]
    nom_list, dists = nominator.predict()
    unique_att = nominator.unique_attributes_
    assert nom_list.shape == (n_verts, unique_att.shape[0])
    assert dists.shape == (n_verts, unique_att.shape[0])
    return nom_list


def _test_seed_input_dimensions():
    with pytest.raises(IndexError):
        _nominate(adj, np.zeros((5, 5, 5), dtype=np.int))


def _test_seed_shape():
    with pytest.raises(IndexError):
        _nominate(adj, np.zeros((1, 50), dtype=np.int))


def _test_seed_input_array_dtype():
    with pytest.raises(TypeError):
        _nominate(adj, np.random.random((10, 2)))


def _test_seed_input_type():
    with pytest.raises(TypeError):
        _nominate(adj, [0] * 10)


def _test_X_input_type():
    with pytest.raises(TypeError):
        _nominate([[0] * 10] * 10, np.zeros(3, dtype=np.int))


def _test_X_array_dtype():
    with pytest.raises(TypeError):
        _nominate(np.random.random((10, 10)), np.zeros(3, dtype=np.int))


def _test_X_input_dimensions():
    with pytest.raises(IndexError):
        _nominate(np.zeros((5, 5, 5), dtype=np.int), np.zeros(3, dtype=np.int))


def _test_embedding_dimensions():
    # embedding should have less cols than rows.
    svn = SpectralVertexNomination(input_graph=False)
    with pytest.raises(IndexError):
        _nominate(
            np.zeros((10, 20), dtype=np.int), np.zeros(3, dtype=np.int), nominator=svn
        )


def _test_adjacency_shape():
    # adj matrix should be square
    with pytest.raises(IndexError):
        _nominate(np.zeros((3, 4), dtype=np.int), np.zeros(3, dtype=np.int))


def _test_input_graph_bool_type():
    svn = SpectralVertexNomination(input_graph=4)
    # input graph param has wrong type
    with pytest.raises(TypeError):
        _nominate(adj, np.zeros(3, dtype=np.int), nominator=svn)


def _test_k_type():
    # k of worng type
    with pytest.raises(TypeError):
        _nominate(adj, np.zeros(3, dtype=np.int), k="hi")


def _test_k_value():
    # k should be > 0
    with pytest.raises(ValueError):
        _nominate(adj, np.zeros(3, dtype=np.int), k=0)


def _test_embedder_type():
    # embedder must be BaseSpectralEmbed or str
    svn = SpectralVertexNomination(embedder=45)
    with pytest.raises(TypeError):
        _nominate(adj, np.zeros(3, dtype=int), nominator=svn)


def _test_embedder_value():
    svn = SpectralVertexNomination(embedder="hi")
    with pytest.raises(ValueError):
        _nominate(adj, np.zeros(3, dtype=int), nominator=svn)


class TestSpectralVertexNominatorOutputs:
    def test_seed_inputs(self):
        _test_seed_input_dimensions()
        _test_seed_shape()
        _test_seed_input_array_dtype()
        _test_seed_input_type()

    def test_X_inputs(self):
        _test_X_array_dtype()
        _test_X_input_dimensions()
        _test_X_input_type()
        _test_embedding_dimensions()
        _test_adjacency_shape()

    def _test_k(self):
        _test_k_value()
        _test_k_type()

    def test_constructor_inputs(self):
        _test_embedder_value()
        _test_embedder_type()
        _test_input_graph_bool_type()

    @pytest.mark.parametrize(
        "nominator",
        [
            SpectralVertexNomination(embedder="ASE"),
            SpectralVertexNomination(embedder="LSE"),
            SpectralVertexNomination(embedder=embeder),
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
    def test_basic_unattributed(self, nominator, seed):
        """
        Runs two attributed seeds and two unattributed seeds with each nominator.
        Ensures all options work. Should be fast. Nested parametrization tests all
        combinations of listed parameters.
        """
        _nominate(adj, seed, nominator)

    @pytest.mark.parametrize(
        "seed",
        [
            np.array([8]),
            np.array([2, 6, 9, 15, 25]),
            _gen_att_seed(10, n_verts, labels),
            _gen_att_seed(20, n_verts, labels),
        ],
    )
    def test_pre_embedded(self, seed):
        svn = SpectralVertexNomination(input_graph=False)
        _nominate(pre_embeded, seed, nominator=svn)

    @pytest.mark.parametrize(
        "nominator",
        [
            SpectralVertexNomination(embedder="ASE"),
            SpectralVertexNomination(embedder="LSE"),
        ],
    )
    @pytest.mark.parametrize(
        "seed", [_gen_att_seed(20, n_verts, labels), _gen_att_seed(50, n_verts, labels)]
    )
    def test_attributed_accuracy(self, nominator, seed):
        """
        want to ensure performance is at least better than random
        """
        group1_correct = np.zeros(3 * n_verts)
        group2_correct = np.zeros(3 * n_verts)
        group3_correct = np.zeros(3 * n_verts)
        labels = np.array([0] * n_verts + [1] * n_verts + [2] * n_verts)
        for i in range(100):
            _adj = np.array(sbm(3 * [n_verts], p), dtype=np.int)
            n_list = _nominate(_adj, seed=seed, nominator=nominator)
            group1_correct[np.argwhere(labels[n_list.T[0]] == 0)] += 1
            group2_correct[np.argwhere(labels[n_list.T[1]] == 1)] += 1
            group3_correct[np.argwhere(labels[n_list.T[2]] == 2)] += 1
        g1_correct_prob = group1_correct / 100
        g2_correct_prob = group2_correct / 100
        g3_correct_prob = group3_correct / 100
        assert np.mean(g1_correct_prob[: n_verts - 10]) > 0.7
        assert np.mean(g2_correct_prob[: n_verts - 10]) > 0.7
        assert np.mean(g3_correct_prob[: n_verts - 10]) > 0.7
