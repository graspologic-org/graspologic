import pytest
from pytest import mark
import numpy as np
from graspologic.embed.case import CovariateAssistedEmbedding as CASE
from graspologic.simulations import sbm
from graspologic.utils import is_almost_symmetric

np.random.seed(420)

# UTILITY FUNCTIONS
def gen_covariates(m1, m2, labels, type, ndim=3):
    n = len(labels)

    if type == "static":
        m1_arr = np.full(n, m1)
        m2_arr = np.full((n, ndim), m2)
        m2_arr[np.arange(n), labels] = m1_arr
    elif type == "normal":
        m1_arr = np.random.choice([1, 0], p=[m1, 1 - m1], size=(n))
        m2_arr = np.random.choice([1, 0], p=[m2, 1 - m2], size=(n, ndim))
        m2_arr[np.arange(n), labels] = m1_arr
    elif type == "many":
        ndim = 300
        gen_covs = lambda size: np.random.choice([1, 0], p=[m1, 1 - m1], size=size)
        m2_arr = np.random.choice([1, 0], p=[m2, 1 - m2], size=(len(labels), ndim))
        m2_arr[labels == 0, :100] = gen_covs(size=m2_arr[labels == 0, :100].shape)
        m2_arr[labels == 1, 100:200] = gen_covs(size=m2_arr[labels == 1, 100:200].shape)
        m2_arr[labels == 2, 200:] = gen_covs(size=m2_arr[labels == 2, 200:].shape)
    else:
        raise ValueError("type must be in ['static', 'normal', 'many']")

    return m2_arr


# FIXTURES
@pytest.fixture(params=[True, False], scope="module")
def M(request):
    # module scope ensures that A and labels will always match
    # since they exist in separate functions

    # parameters
    n = 5
    p, q = 0.9, 0.3

    # block probability matirx
    P = np.full((2, 2), q)
    P[np.diag_indices_from(P)] = p

    # generate sbm
    directed = request.param
    return sbm([n] * 3, P, directed=directed, return_labels=True)


@pytest.fixture(params=["static", "normal", "many"])
def X(request, M):
    _, labels = M
    m1, m2 = 0.8, 0.3
    return gen_covariates(m1, m2, labels, type=request.param)


@pytest.fixture
def case(M, X):
    A, labels = M
    case = CASE(n_components=2)
    case.fit(A, covariates=X)
    return case


@pytest.fixture
def A(M):
    return M[0]


@pytest.fixture
def labels(M):
    return M[1]


# TESTS
def test_case_fits(case):
    assert case


def test_wrong_inputs(A, X):

    with pytest.raises(TypeError):
        "wrong assortive type"
        case = CASE(assortative=1)

    with pytest.raises(ValueError):
        "without covariates"
        CASE().fit(A)


def test_bad_covariate_matrix(X):
    "not an adjacency matrix"
    with pytest.raises(ValueError):
        A_ = np.arange(30).reshape(10, 3)
        CASE().fit(A_, X)


def test_fit_transform(A, X):
    case = CASE(n_components=2)
    assert case.fit_transform(A, covariates=X).any()


def test_labels_match_clustering(M, case):
    if np.array_equal(X, X.astype(bool)) and X.shape[-1] == 3:
        # We already know binary covariates with small dimensionality
        # make things weird
        pytest.skip()
    A, labels = M


def test_directed_correct_latent_positions():
    # setup
    ase = AdjacencySpectralEmbed(n_components=3)
    P = np.array([[0.9, 0.1, 0.1], [0.3, 0.6, 0.1], [0.1, 0.5, 0.6]])
    M, labels = sbm([200, 200, 200], P, directed=True, return_labels=True)

    # one node from each community
    oos_idx = np.nonzero(np.r_[1, np.diff(labels)[:-1]])[0]
    labels = list(labels)
    oos_labels = [labels.pop(i) for i in oos_idx]

    # Grab out-of-sample, fit, transform
    A, a = remove_vertices(M, indices=oos_idx, return_removed=True)
    latent_left, latent_right = ase.fit_transform(A)
    oos_left, oos_right = ase.transform(a)

    # separate into communities
    for i, latent in enumerate([latent_left, latent_right]):
        left = i == 0
        df = pd.DataFrame(
            {
                "Type": labels,
                "Dimension 1": latent[:, 0],
                "Dimension 2": latent[:, 1],
                "Dimension 3": latent[:, 2],
            }
        )
        # make sure that oos vertices are closer to their true community averages than other community averages
        means = df.groupby("Type").mean()
        if left:
            avg_dist_within = np.diag(pairwise_distances(means, oos_left))
            avg_dist_between = np.diag(pairwise_distances(means, oos_right))
            self.assertTrue(all(avg_dist_within < avg_dist_between))
        elif not left:
            avg_dist_within = np.diag(pairwise_distances(means, oos_right))
            avg_dist_between = np.diag(pairwise_distances(means, oos_left))
            self.assertTrue(all(avg_dist_within < avg_dist_between))


def test_covariates_improve_clustering(M):
    if np.array_equal(X, X.astype(bool)) and X.shape[-1] == 3:
        # We already know binary covariates with small dimensionality
        # make things weird
        pytest.skip()
    A, labels = M
    pass


class TestGenCovariates:
    def test_gen_covariates_determined(self, m1=1.0, m2=0.0):
        # basic test on an identity matrix with 100% probabilities
        labels = np.array([0, 1, 2])
        X = gen_covariates(m1, m2, labels=labels, type="normal")
        assert np.array_equal(X, np.eye(3))

    def test_gen_covariates_determined_repeated(self, m1=1.0, m2=0.0):
        # test on a repeated identity matrix with 100% probabilities
        labels = np.repeat(np.array([0, 1, 2]), repeats=3)
        I = np.repeat(np.eye(3), repeats=3, axis=0)
        X = gen_covariates(m1, m2, labels=labels, type="normal")
        assert np.array_equal(X, I)
