import pytest
from pytest import mark
import numpy as np
from sklearn.mixture import GaussianMixture
from graspologic.embed.case import CovariateAssistedEmbed as CASE
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
@pytest.fixture(params=[False], scope="module")
def M(request):
    # module scope ensures that A and labels will always match
    # since they exist in separate functions

    # parameters
    n = 10
    p, q = 0.9, 0.3

    # block probability matrix
    P = np.full((2, 2), q)
    P[np.diag_indices_from(P)] = p

    # generate sbm
    directed = request.param
    return sbm([n] * 2, P, directed=directed, return_labels=True)


@pytest.fixture(params=["static", "many"])
def X(request, M):
    _, labels = M
    m1, m2 = 0.8, 0.3
    return gen_covariates(m1, m2, labels, type=request.param)


@pytest.fixture(params=["assortative", "non-assortative"])
def case(request, M, X):
    A, _ = M
    case = CASE(n_components=2, embedding_alg=request.param)
    case.fit((A, X))
    return case


@pytest.fixture
def A(M):
    return M[0]


@pytest.fixture
def labels(M):
    return M[1]


# TESTS
@pytest.mark.parametrize("alg", ["assortative", "non-assortative"])
def test_custom_alpha(M, X, alg):
    A, _ = M
    a = 0.1
    case = CASE(n_components=2, embedding_alg=alg, alpha=a)
    latents = case.fit_transform((A, X))
    # just check to make sure we can run with custom alpha

    if case.alpha_ == 0.1:
        assert np.any(latents)
    else:
        pytest.fail("no alpha value", pytrace=True)


def test_case_fits(case):
    assert case


def test_labels_match(A, labels, M):
    A_, labels_ = M
    assert np.array_equal(A, A_)
    assert np.array_equal(labels, labels_)


def test_wrong_inputs(A, X):

    with pytest.raises(ValueError):
        "wrong assortative type"
        CASE(embedding_alg=1)

    with pytest.raises(ValueError):
        A_ = np.arange(30).reshape(10, 3)
        CASE().fit((A_, X))


def test_fit_transform(A, X):
    case = CASE(n_components=2)
    directed = not is_almost_symmetric(A)
    if directed:
        assert case.fit_transform((A, X))[0].any()
    else:
        assert case.fit_transform((A, X)).any()


def test_labels_match_clustering(case, labels):
    """
    train a GMM, assert predictions match labels
    """
    # should get 100% accuracy since the two clusters are super different
    latent = case.latent_left_
    gmm = GaussianMixture(n_components=2)
    predicted_labels = gmm.fit_predict(latent)
    assert np.array_equal(predicted_labels, labels) or np.array_equal(
        1 - predicted_labels, labels
    )


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
