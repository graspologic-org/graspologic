#!/usr/bin/env python

# embed.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

from sklearn.decomposition import TruncatedSVD


def selectDim(X, method=TruncatedSVD, *args, **kwargs):
    """
    A function to use Ghodsi et al. for dimensionality selection with
    the SVD.

    Inputs:
    -------
        X: {array-like}, shape (n_samples, n_features)
         The input data to select the optimal embedding dimensionality for.
        method: object (default TruncatedSVD)
        args: list, optional (default None)
         options taken by the desired embedding method as arguments.
        kwargs: dict, optional (default None)
         options taken by the desired embedding method as key-worded
         arguments.

    Returns:
    --------
        optimal_d: {int}
         the optimal number of embedding dimensions.
        optimal_lq: {float}
         the likelihood of the optimal number of embedding dimensions.
        ds: {array-like}, shape (n_components)
         the singular values associated with the decomposition of X,
         from which optimal_d was chosen.
        lqs: {array-like}, shape (n_components)
         the likelihood profile for all possible embedding dimensions
         ds.
    See Also:
    ---------
        TruncatedSVD
    """

    return {'optimal_d': optimal_d, 'optimal_lq': optimal_lq,
            'ds': ds, 'lqs': lqs}

def SelectSVD(TruncatedSVD):
    """
    A class that wraps Scikit-Learn's sklearn.decomposition.SVD method.

    Dimensionality reduction using truncated SVD (aka LSA).
    This transformer performs linear dimensionality reduction by means of
    truncated singular value decomposition (SVD). Contrary to PCA, this
    estimator does not center the data before computing the singular value
    decomposition. This means it can work with scipy.sparse matrices
    efficiently.
    In particular, truncated SVD works on term count/tf-idf matrices as
    returned by the vectorizers in sklearn.feature_extraction.text. In that
    context, it is known as latent semantic analysis (LSA).
    This estimator supports two algorithms: a fast randomized SVD solver, and
    a "naive" algorithm that uses ARPACK as an eigensolver on (X * X.T) or
    (X.T * X), whichever is more efficient.
    Read more in the :ref:`User Guide <LSA>`.
    Parameters
    ----------
    n_components : int, default = 2
        Desired dimensionality of output data.
        Must be strictly less than the number of features.
        The default value is useful for visualisation. For LSA, a value of
        100 is recommended.
    algorithm : string, default = "randomized"
        SVD solver to use. Either "arpack" for the ARPACK wrapper in SciPy
        (scipy.sparse.linalg.svds), or "randomized" for the randomized
        algorithm due to Halko (2009).
    n_iter : int, optional (default 5)
        Number of iterations for randomized SVD solver. Not used by ARPACK.
        The default is larger than the default in `randomized_svd` to handle
        sparse matrices that may have large slowly decaying spectrum.
    random_state : int, RandomState instance or None, optional, default = None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    tol : float, optional
        Tolerance for ARPACK. 0 means machine precision. Ignored by randomized
        SVD solver.
    Attributes
    ----------
    components_ : array, shape (n_components, n_features)
    explained_variance_ : array, shape (n_components,)
        The variance of the training samples transformed by a projection to
        each component.
    explained_variance_ratio_ : array, shape (n_components,)
        Percentage of variance explained by each of the selected components.
    singular_values_ : array, shape (n_components,)
        The singular values corresponding to each of the selected components.
        The singular values are equal to the 2-norms of the ``n_components``
        variables in the lower-dimensional space.
    Examples
    --------
    >>> from sklearn.decomposition import TruncatedSVD
    >>> from sklearn.random_projection import sparse_random_matrix
    >>> X = sparse_random_matrix(100, 100, density=0.01, random_state=42)
    >>> svd = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    >>> svd.fit(X)  # doctest: +NORMALIZE_WHITESPACE
    TruncatedSVD(algorithm='randomized', n_components=5, n_iter=7,
            random_state=42, tol=0.0)
    >>> print(svd.explained_variance_ratio_)  # doctest: +ELLIPSIS
    [0.0606... 0.0584... 0.0497... 0.0434... 0.0372...]
    >>> print(svd.explained_variance_ratio_.sum())  # doctest: +ELLIPSIS
    0.249...
    >>> print(svd.singular_values_)  # doctest: +ELLIPSIS
    [2.5841... 2.5245... 2.3201... 2.1753... 2.0443...]
    See also
    --------
    sklearn.decomposition.PCA, sklearn.decomposition.TruncatedSVD
    References
    ----------
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) http://arxiv.org/pdf/0909.4061
    Notes
    -----
    SVD suffers from a problem called "sign indeterminacy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.
    """
    def __init__(self, n_components=2, algorithm="randomized", n_iter=5,
                 random_state=None, tol=0.):
        self.algorithm = algorithm
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol

    def fit(self, X, y=None):
        """Fit LSI model on training data X.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Training data.
        y : Ignored
        Returns
        -------
        self : object
            Returns the transformer object.
        """
        # TODO: comment out below to add dynamic dimensionality selection.
        #if self.n_components is None:
        #	self.n_components = selectDim(X)
        self.fit_transform(X)
        return self
