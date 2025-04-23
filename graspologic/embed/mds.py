# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array
from typing_extensions import Literal

from ..utils import is_symmetric
from .svd import SvdAlgorithmType, select_svd

if TYPE_CHECKING:
    from graspologic.types import Tuple


def _get_centering_matrix(n: int) -> np.ndarray:
    """
    Compute the centering array

    Parameters
    ----------
    n : int
        Size of centering array.

    Returns
    -------
    out : 2d-array
        Outputs a centering array of shape (n, n)
    """
    out = np.identity(n) - (1 / n) * np.ones((n, n))

    return out


class ClassicalMDS(BaseEstimator):
    """
    Classical multidimensional scaling (cMDS).

    cMDS seeks a low-dimensional representation of the data in which the distances
    respect well the distances in the original high-dimensional space.

    Parameters
    ----------
    n_components : int, or None (default=None)
        Number of components to keep. If None, then it will run
        :func:`~graspologic.embed.select_dimension` to find the optimal embedding dimension.

    n_elbows : int, or None (default=2)
        If ``n_components`` is None, then compute the optimal embedding dimension using
        :func:`~graspologic.embed.select_dimension`. Otherwise, ignored.

    dissimilarity : 'euclidean' | 'precomputed', optional, default: 'euclidean'
        Dissimilarity measure to use:

        - 'euclidean'
            Pairwise Euclidean distances between points in the dataset.
        - 'precomputed'
            Pre-computed dissimilarities are passed directly to :func:`~graspologic.embed.ClassicalMDS.fit` and
            :func:`~graspologic.embed.ClassicalMDS.fit_transform`.

    Attributes
    ----------
    n_components_ : int
        Equals the parameter ``n_components``. If input ``n_components`` was None,
        then equals the optimal embedding dimension.

    n_features_in_: int
        Number of features passed to the :func:`~graspologic.embed.ClassicalMDS.fit` method.

    components_ : array, shape (n_components, n_features)
        Principal axes in feature space.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.

    dissimilarity_matrix_ : array, shape (n_features, n_features)
        Dissimilarity matrix

    svd_seed : int or None (default ``None``)
        Only applicable for ``n_components!=1``; allows you to seed the
        randomized svd solver for deterministic, albeit pseudo-randomized behavior.

    See Also
    --------
    graspologic.embed.select_dimension

    References
    ----------
    Wickelmaier, Florian. "An introduction to MDS." Sound Quality Research Unit,
    Aalborg University, Denmark 46.5 (2003).
    """

    def __init__(
        self,
        n_components: Optional[int] = None,
        n_elbows: int = 2,
        dissimilarity: Literal["euclidean", "precomputed"] = "euclidean",
        svd_seed: Optional[int] = None,
    ) -> None:
        # Set properties before error checking
        self.dissimilarity = dissimilarity

        self.n_elbows = n_elbows
        self.svd_seed = svd_seed
        self.n_components = n_components

    def _compute_euclidean_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Computes pairwise distance between row vectors or matrices

        Parameters
        ----------
        X : array_like
            If ``dissimilarity=='precomputed'``, the input should be the
            dissimilarity matrix with shape (n_samples, n_samples). If
            ``dissimilarity=='euclidean'``, then the input should be 2d-array
            with shape (n_samples, n_features) or a 3d-array with shape
            (n_samples, n_features_1, n_features_2).

        Returns
        -------
        out : 2d-array, shape (n_samples, n_samples)
            A dissimilarity matrix based on Frobenous norms between pairs of
            matrices or vectors.
        """
        shape = X.shape
        n_samples = shape[0]

        order: Literal[2, "fro"]
        axis: Union[int, Tuple[int, int]]
        if X.ndim == 2:
            order = 2
            axis = 1
        else:
            order = "fro"
            axis = (1, 2)

        out = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            out[i] = np.linalg.norm(X - X[i], axis=axis, ord=order)

        return out

    def fit(self, X: np.ndarray, y: Optional[Any] = None) -> "ClassicalMDS":
        """
        Fit the model with X.

        Parameters
        ----------
        X : array_like
            If ``dissimilarity=='precomputed'``, the input should be the
            dissimilarity matrix with shape (n_samples, n_samples). If
            ``dissimilarity=='euclidean'``, then the input should be 2d-array
            with shape (n_samples, n_features) or a 3d-array with shape
            (n_samples, n_features_1, n_features_2).

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        # Check inputs
        # ScikitLearn moved validations to fit(). See https://github.com/scikit-learn/scikit-learn/issues/30667
        if self.n_components is not None:
            if not isinstance(self.n_components, int):
                msg = "n_components must be an integer, not {}.".format(
                    type(self.n_components)
                )
                raise TypeError(msg)
            elif self.n_components <= 0:
                msg = "n_components must be >= 1 or None."
                raise ValueError(msg)

        if self.dissimilarity not in ["euclidean", "precomputed"]:
            msg = "Dissimilarity measure must be either 'euclidean' or 'precomputed'."
            raise ValueError(msg)
        # Check X type
        if not isinstance(X, np.ndarray):
            msg = "X must be a numpy array, not {}.".format(type(X))
            raise ValueError(msg)

        if self.n_components is not None:
            n_samples = X.shape[0]
            if self.n_components > n_samples:
                msg = "n_components must be <= n_samples."
                raise ValueError(msg)

        # Handle dissimilarity
        if self.dissimilarity == "precomputed":
            dissimilarity_matrix = check_array(X, ensure_2d=True, allow_nd=False)

            # Must be symmetric
            if not is_symmetric(dissimilarity_matrix):
                msg = "X must be a symmetric array if precomputed dissimilarity matrix."
                raise ValueError(msg)
        elif self.dissimilarity == "euclidean":
            X = check_array(X, ensure_2d=True, allow_nd=True)
            dissimilarity_matrix = self._compute_euclidean_distances(X=X)

        J = _get_centering_matrix(dissimilarity_matrix.shape[0])
        B = J @ (dissimilarity_matrix**2) @ J * -0.5

        n_components = self.n_components

        algorithm: SvdAlgorithmType
        if n_components == 1:
            algorithm = "full"
        else:
            algorithm = "randomized"
        U, D, V = select_svd(
            B,
            n_elbows=self.n_elbows,
            algorithm=algorithm,
            n_components=n_components,
            svd_seed=self.svd_seed,
        )

        self.n_components_ = len(D)
        self.components_ = U
        self.singular_values_ = D**0.5
        self.dissimilarity_matrix_ = dissimilarity_matrix
        self.n_features_in_ = X.shape[1]

        return self

    def fit_transform(self, X: np.ndarray, y: Optional[Any] = None) -> np.ndarray:
        """
        Fit the data from X, and returns the embedded coordinates.

        Parameters
        ----------
        X : nd-array
            If ``dissimilarity=='precomputed'``, the input should be the
            dissimilarity matrix with shape (n_samples, n_samples). If
            ``dissimilarity=='euclidean'``, then the input should be array
            with shape (n_samples, n_features) or a nd-array with shape
            (n_samples, n_features_1, n_features_2, ..., n_features_d). First
            axis of nd-array must be ``n_samples``.

        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
            Embedded input.
        """
        self.fit(X)

        X_new = self.components_ @ np.diag(self.singular_values_)

        return X_new
