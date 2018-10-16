import pytest
import numpy as np
from numpy.testing import assert_almost_equal

from graspy.embed.mds import ClassicalMDS


def test_input():
    X = np.random.normal(0, 1, size=(10, 3))

    # n_components > n_samples
    with pytest.raises(ValueError):
        mds = ClassicalMDS(n_components=100)
        mds.fit(X)

    # Invalid n_components
    with pytest.raises(ValueError):
        mds = ClassicalMDS(n_components=-2)

    with pytest.raises(TypeError):
        mds = ClassicalMDS(n_components='1')

    # Invalid dissimilarity
    with pytest.raises(ValueError):
        mds = ClassicalMDS(dissimilarity='abc')

    # n_components > n_samples / n_elements
    with pytest.raises(ValueError):
        mds = ClassicalMDS(n_components=6)
        mds.fit(X, n_elements=2)

    # Must be square and symmetric matrix if precomputed dissimilarity
    with pytest.raises(ValueError):
        mds = ClassicalMDS(n_components=3, dissimilarity='precomputed')
        mds.fit(X)


def test_output():
    """
    Recover a 3D tetrahedron with distance 1 between all points

    Use both fit and fit_transform functions
    """

    def _compute_dissimilarity(arr):
        out = np.zeros((4, 4))

        for i in range(4):
            out[i] = np.linalg.norm(arr - arr[i], axis=1)

        return out

    def use_fit_transform():
        A = np.ones((4, 4)) - np.identity(4)

        mds = ClassicalMDS(n_components=3, dissimilarity='precomputed')
        B = mds.fit_transform(A)

        Ahat = _compute_dissimilarity(B)

        # Checks up to 7 decimal points
        assert_almost_equal(A, Ahat)

    def use_fit():
        A = np.ones((4, 4)) - np.identity(4)

        mds = ClassicalMDS(n_components=3, dissimilarity='precomputed')
        mds.fit(A)
        B = np.dot(mds.components_, np.diag(mds.singular_values_))

        Ahat = _compute_dissimilarity(B)

        # Checks up to 7 decimal points
        assert_almost_equal(A, Ahat)

    use_fit_transform()
    use_fit()