import numpy as np
from numpy.testing import assert_almost_equal

from graphstats.embed.mds import ClassicalMDS


def test_output():
    """
    Recover a 3D tetrahedron with distance 1 between all points
    """

    def _compute_dissimilarity(arr):
        out = np.zeros((4, 4))

        for i in range(4):
            out[i] = np.linalg.norm(arr - arr[i], axis=1)

        return out

    A = np.ones((4, 4)) - np.identity(4)

    mds = ClassicalMDS(n_components=3)
    B = mds.fit_transform(A)

    Ahat = _compute_dissimilarity(B)

    # Checks up to 7 decimal points
    assert_almost_equal(A, Ahat)