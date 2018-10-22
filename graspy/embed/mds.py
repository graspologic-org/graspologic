import numpy as np
from sklearn.base import BaseEstimator

from .svd import selectSVD
from ..utils import is_symmetric


def _get_centering_matrix(n):
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

    cMDS  seeks a low-dimensional representation of the data in
    which the distances respect well the distances in the original
    high-dimensional space.

    Parameters
    ----------
    n_components : int, or None
        Number of components to keep. If None, then it will run
        ``dimselect`` to find the optimal embedding dimension.
    
    dissimilarity : 'euclidean' | 'precomputed', optional, default: 'euclidean'
        Dissimilarity measure to use:

        - 'euclidean':
            Pairwise Euclidean distances between points in the dataset.

        - 'precomputed':
            Pre-computed dissimilarities are passed directly to ``fit`` and
            ``fit_transform``.

    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
        Principal axes in feature space.

    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.

    n_components : int
        Equals the parameter n_components, or n_features if n_components
        is None.

    dissimilarity_matrix_ : array, shape (n_features, n_features)
        Dissimilarity matrix 

    See Also
    --------
    graphstats.embed.dimselect
    """

    def __init__(self, n_components=None, dissimilarity='euclidean'):
        # Check inputs
        if n_components is not None:
            if not isinstance(n_components, int):
                msg = "n_components must be an integer, not {}.".format(
                    type(n_components))
                raise TypeError(msg)
            elif n_components <= 0:
                msg = "n_components must be >= 1 or None."
                raise ValueError(msg)
        self.n_components = n_components

        if dissimilarity not in ['euclidean', 'precomputed']:
            msg = "Dissimilarity measure must be either 'euclidean' or 'precomputed'."
            raise ValueError(msg)
        self.dissimilarity = dissimilarity

    def _compute_euclidean_distances(self, X):
        """
        Computes pairwise distance between row vectors or matrices

        Parameters
        ----------
        X : nd-array
            If ``dissimilarity=='precomputed'``, the input should be the 
            dissimilarity matrix with shape (n_samples, n_samples). If 
            ``dissimilarity=='euclidean'``, then the input should be array 
            with shape (n_samples, n_features) or a nd-array with shape 
            (n_samples, n_features_1, n_features_2, ..., n_features_d).
        
        Returns
        -------
        out : 2d-array, shape (n_samples // n_elements, n_samples // n_elements)
            A dissimilarity matrix based on Frobenous norms between pairs of
            matrices or vectors.
        """
        shape = X.shape
        n_samples = shape[0]

        out = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            out[i] = np.linalg.norm(X - X[i], axis=(1, 2), ord='fro')

        return out

    def fit(self, X):
        """
        Fit the model with X.

        Parameters
        ----------
        X : nd-array
            If ``dissimilarity=='precomputed'``, the input should be the 
            dissimilarity matrix with shape (n_samples, n_samples). If 
            ``dissimilarity=='euclidean'``, then the input should be array 
            with shape (n_samples, n_features) or a nd-array with shape 
            (n_samples, n_features_1, n_features_2, ..., n_features_d).
        """
        # Check X type
        if not isinstance(X, np.ndarray):
            msg = "X must be a numpy array, not {}.".format(type(X))
            raise ValueError(msg)

        n_samples = X.shape[0]
        # Handle n_components
        if n_samples <= self.n_components:
            msg = "n_components must be <= (n_samples / n_elements)."
            raise ValueError(msg)
        elif self.n_components is None:
            # TODO: use dimselect here
            n_components = min(dissimilarity_matrix.shape) - 1
        else:
            n_components = self.n_components

        # Handle dissimilarity
        if self.dissimilarity == 'precomputed':
            # Handle shape of X if precomputed distance matrix
            if len(X.shape) != 2:
                msg = "X must be a 2d-array. Input has {} dimensions.".format(
                    len(X.shape))
                raise ValueError(msg)
            # Must be symmetric
            if not is_symmetric(X):
                msg = "X must be a symmetric array if precomputed dissimilarity matrix."
                raise ValueError(msg)
            dissimilarity_matrix = X
        elif self.dissimilarity == 'euclidean':
            dissimilarity_matrix = self._compute_euclidean_distances(X=X)

        J = _get_centering_matrix(dissimilarity_matrix.shape[0])
        B = J.dot(dissimilarity_matrix**2).dot(J) * -0.5

        U, V, D = selectSVD(B, k=n_components)

        self.components_ = U
        self.singular_values_ = D**0.5
        self.dissimilarity_matrix_ = dissimilarity_matrix

        return self

    def fit_transform(self, X):
        """
        Fit the model with X and apply the dimensionality reduction on X.

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
        X_new : array-like 
        """
        self.fit(X)

        X_new = np.dot(self.components_, np.diag(self.singular_values_))

        return X_new
