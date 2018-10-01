#!/usr/bin/env python

# svd.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

from sklearn.decomposition import TruncatedSVD
from scipy.sparse.linalg import svds
import numpy as np
from scipy.stats import norm

<<<<<<< HEAD
def selectDim(data, n_elbows=1, threshold=0):
=======

def selectDim(X, method=TruncatedSVD, *args, **kwargs):
>>>>>>> 4b0389aba0a0ef80740fff8fc2049513654c5247
    """
    Generates profile likelihood from array based on Z&G.

    Returns an array of elbows and relevant information

    Parameters
    ----------
    data : array_like
        The matrix of data we are trying to generate profile likelihoods for.
    n_elbows : int, optional
        Number of likelihood elbows to return.
    method : object, optional
        Takes an object to calculate the svd

    Returns
    -------
    elbows : array_like
        ZG elbows which indicate subsequent optimal embedding dimensions.
    likelihoods : array_like
        Array of likelihoods of the optimal embedding dimensions.
    sing_vals : array_like
        The singular values of the data array post-threshold.
    all_likelihoods : array_like
        The likelihood profiles of all embedding dimensions.

    Other Parameters
    ----------------
    threshold : float, optional
        Ignores eigenvalues smaller than this.

    Raises
    ------
    ValueError
        If n_elbows is :math:`< 1`.

    References
    ----------
    .. [1] Zhu, M. and Ghodsi, A. (2006).
        Automatic dimensionality selection from the scree plot via the use of profile likelihood.
        Computational Statistics & Data Analysis, 51(2), pp.918-930.

    """
    if n_elbows < 1:
        msg = 'number of elbows should be an integer > 1, not {}'
        raise ValueError(msg.format(n_elbows))

    # generate eigenvalues greater than the threshold
    sing_vals = svds(data, k=min(data.shape)-1, return_singular_vectors=False)[::-1]
    L = sing_vals**2
    L2 = L[L > threshold]
    U = L2
    if L.ndim == 2:
        L = np.std(U, axis=0)

    if len(U) == 0:
        msg = 'no eigenvalues ({}) greater than threshold {}'
        raise IndexError(msg.format(L, threshold))

    elbows = []
    if len(U) == 1:
        return np.array(elbows.append(U[0]))+1

    n = len(U)
    all_l = []
    elbow_l = []
    while len(elbows) < n_elbows and len(U) > 1:
        d = 1
        sample_var = np.var(U, ddof=1)
        sample_scale = sample_var**(1/2)
        elbow = 0
        likelihood_elbow = 0
        l = []
        while d < len(U):
            mean_sig = np.mean(U[:d])
            mean_noise = np.mean(U[d:])
            sig_likelihood = 0
            noise_likelihood = 0
            for i in range(d):
                sig_likelihood += norm.pdf(U[i], mean_sig, sample_scale)
            for i in range(d, len(U)):
                noise_likelihood += norm.pdf(U[i], mean_noise, sample_scale)

            likelihood = noise_likelihood + sig_likelihood
            l.append(likelihood)

            if likelihood > likelihood_elbow:
                likelihood_elbow = likelihood
                elbow_l.append(likelihood)
                elbow = d
            d += 1
        if len(elbows) == 0:
            elbows.append(elbow)
        else:
            elbows.append(elbow + elbows[-1])
        U = U[elbow:]
        all_l.append(l)

    if len(elbows) == n_elbows:
        return np.array(elbows)+1, elbow_l, L2, all_l

    if len(U) == 0:
        return np.array(elbows)+1, elbow_l, L2, all_l
    else:
        elbows.append(n)
        return np.array(elbows)+1, elbow_l, L2, all_l


def selectSVD(X, k=None):
    """
    A function for performing svd using ZG2, X = U S Vt.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        the data to perform svd on.
    k: int
        the number of dimensions to embed into. Should have
        k < min(X.shape).

    Returns
    -------
    U: array-like, shape (n_samples, k)
        the left singular vectors.
    V: array-like, shape (n_samples, k)
        the right singular vectors.
    s: array-like, shape (k)
        the singular values, as a 1d array.
    """
    if (k is None):
        selectDim(X)
    if k > min(X.shape):
        msg = "k is {}, but min(X.shape) is {}."
        msg = msg.format(k, min(X.shape))
        raise ValueError(msg)
    U, s, Vt = svds(X, k=k)
    return (U, Vt.T, s)
<<<<<<< HEAD


class SelectSVD(TruncatedSVD):
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

    See also
    --------
    sklearn.decomposition.PCA, sklearn.decomposition.TruncatedSVD
    References
    ----------
    Finding structure with randomness: Stochastic algorithms for constructing
    approximate matrix decompositions
    Halko, et al., 2009 (arXiv:909) http://arxiv.org/pdf/0909.4061

    Automatic dimensionality selection from the scree plot via the use of
    profile likelihood
    Zhu, Mu and Ghodsi, Ali. CSDA 2006.
    https://www.sciencedirect.com/science/article/pii/S0167947305002343

    Notes
    -----
    SVD suffers from a problem called "sign indeterminacy", which means the
    sign of the ``components_`` and the output from transform depend on the
    algorithm and random state. To work around this, fit instances of this
    class to data once, then keep the instance around to do transformations.
    """

    def __init__(self,
                 n_components=2,
                 algorithm="randomized",
                 n_iter=5,
                 random_state=None,
                 tol=0.):
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
        #   self.n_components = selectDim(X)
        self.fit_transform(X)
        return self
=======
>>>>>>> 4b0389aba0a0ef80740fff8fc2049513654c5247
