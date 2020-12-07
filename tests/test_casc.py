import pytest
import numpy as np
from graspologic.embed.casc import CASC, gen_covariates
from graspologic.simulations import sbm

np.random.seed(420)

# FIXTURES
@pytest.fixture(params=[True, False])
def M(request):
    # parameters
    n = 5
    p, q = 0.9, 0.3

    # block probability matirx
    P = np.full((3, 3), q)
    P[np.diag_indices_from(P)] = q

    # generate sbm
    directed = request.param
    return sbm([n] * 3, P, directed=directed, return_labels=True)


# TESTS
class TestGenCovariates:
    def test_gen_covariates_determined(self, m1=1.0, m2=0.0):
        # basic test on an identity matrix with 100% probabilities
        labels = np.array([0, 1, 2])
        X = gen_covariates(m1, m2, labels=labels)
        assert np.array_equal(X, np.eye(3))

    def test_gen_covariates_determined_repeated(self, m1=1.0, m2=0.0):
        # test on a repeated identity matrix with 100% probabilities
        labels = np.repeat(np.array([0, 1, 2]), repeats=3)
        I = np.repeat(np.eye(3), repeats=3, axis=0)
        X = gen_covariates(m1, m2, labels=labels)
        assert np.array_equal(X, I)


class TestCasc:
    def __init__(self, M):
        self.A, self.labels = M

    def test_fits(self):
        # make sure CASC fits on our M matrix
        casc = CASC(n_components=2)
        casc.fit(self.A)
        assert casc

    def test_wrong_types(self):
        pass

    def test_wrong_values(self):
        pass

    def test_no_covariate_matrix(self):
        pass

    def test_bad_covariate_matrix(self):
        pass

    def test_labels_match_clustering(self):
        pass

    def test_covariates_improve_clustering(self):
        pass