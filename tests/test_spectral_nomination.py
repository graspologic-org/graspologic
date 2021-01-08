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


def _nominate(X, seed, nominator=None, k=None):
    if nominator is None:
        nominator = SpectralVertexNomination(n_neighbors=k)
    nominator.fit(X)
    n_verts = X.shape[0]
    nom_list, dists = nominator.predict(seed)
    assert nom_list.shape == (n_verts, seed.shape[0])
    assert dists.shape == (n_verts, seed.shape[0])
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
    # input graph param has wrong type
    with pytest.raises(TypeError):
        svn = SpectralVertexNomination(input_graph=4)


def _test_k_type():
    # k of wrong type
    with pytest.raises(TypeError):
        _nominate(adj, np.zeros(3, dtype=np.int), k="hello world")


def _test_k_value():
    # k should be > 0
    with pytest.raises(ValueError):
        _nominate(adj, np.zeros(3, dtype=np.int), k=0)


def _test_embedder_type():
    # embedder must be BaseSpectralEmbed or str
    with pytest.raises(TypeError):
        svn = SpectralVertexNomination(embedder=45)


def _test_embedder_value():
    with pytest.raises(ValueError):
        svn = SpectralVertexNomination(embedder="hi")
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

    def test_constructor_inputs1(self):
        _test_embedder_type()

    def test_constructor_inputs2(self):
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
            np.arange(n_verts - 1, dtype=np.int),
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
            np.arange(n_verts - 1, dtype=np.int),
        ],
    )
    def test_pre_embedded(self, seed):
        svn = SpectralVertexNomination(input_graph=False)
        _nominate(pre_embeded, seed, nominator=svn)
