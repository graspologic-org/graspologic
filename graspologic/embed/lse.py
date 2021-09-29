# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.


from typing import Any, Optional, Union

import networkx as nx
import numpy as np

from ..types import GraphRepresentation
from ..utils import LaplacianFormType, to_laplacian
from .base import BaseSpectralEmbed, SvdAlgorithmType


class LaplacianSpectralEmbed(BaseSpectralEmbed):
    r"""
    Class for computing the laplacian spectral embedding of a graph.

    The laplacian spectral embedding (LSE) is a k-dimensional Euclidean representation
    of the graph based on its Laplacian matrix. It relies on an SVD to reduce
    the dimensionality to the specified ``n_components``, or if ``n_components`` is
    unspecified, can find a number of dimensions automatically.

    Parameters
    ----------
    form : {'DAD' (default), 'I-DAD', 'R-DAD'}, optional
        Specifies the type of Laplacian normalization to use. See
        :func:`~graspologic.utils.to_laplacian` for more details regarding form.

    n_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        ``n_components`` must be ``<= min(X.shape)``. Otherwise, ``n_components`` must be
        ``< min(X.shape)``. If None, then optimal dimensions will be chosen by
        :func:`~graspologic.embed.select_dimension` using ``n_elbows`` argument.

    n_elbows : int, optional, default: 2
        If ``n_components`` is None, then compute the optimal embedding dimension using
        :func:`~graspologic.embed.select_dimension`. Otherwise, ignored.

    algorithm : {'randomized' (default), 'full', 'truncated'}, optional
        SVD solver to use:

        - 'randomized'
            Computes randomized svd using
            :func:`sklearn.utils.extmath.randomized_svd`
        - 'full'
            Computes full svd using :func:`scipy.linalg.svd`
        - 'truncated'
            Computes truncated svd using :func:`scipy.sparse.linalg.svds`

    n_iter : int, optional (default = 5)
        Number of iterations for randomized SVD solver. Not used by 'full' or
        'truncated'. The default is larger than the default in randomized_svd
        to handle sparse matrices that may have large slowly decaying spectrum.

    check_lcc : bool , optional (defult = True)
        Whether to check if input graph is connected. May result in non-optimal
        results if the graph is unconnected. If True and input is unconnected,
        a UserWarning is thrown. Not checking for connectedness may result in
        faster computation.

    regularizer: int, float or None, optional (default=None)
        Constant to be added to the diagonal of degree matrix. If None, average
        node degree is added. If int or float, must be >= 0. Only used when
        ``form`` is 'R-DAD'.

    concat : bool, optional (default False)
        If graph is directed, whether to concatenate left and right (out and in) latent
        positions along axis 1.


    Attributes
    ----------
    n_features_in_: int
        Number of features passed to the
        :func:`~graspologic.embed.LaplacianSpectralEmbed.fit` method.

    latent_left_ : array, shape (n_samples, n_components)
        Estimated left latent positions of the graph.

    latent_right_ : array, shape (n_samples, n_components), or None
        Only computed when the graph is directed, or adjacency matrix is assymetric.
        Estimated right latent positions of the graph. Otherwise, None.

    singular_values_ : array, shape (n_components)
        Singular values associated with the latent position matrices.

    svd_seed : int or None (default ``None``)
        Only applicable for ``algorithm="randomized"``; allows you to seed the
        randomized svd solver for deterministic, albeit pseudo-randomized behavior.

    See Also
    --------
    graspologic.embed.select_svd
    graspologic.embed.select_dimension
    graspologic.utils.to_laplacian

    Notes
    -----
    The singular value decomposition:

    .. math:: A = U \Sigma V^T

    is used to find an orthonormal basis for a matrix, which in our case is the
    Laplacian matrix of the graph. These basis vectors (in the matrices U or V) are
    ordered according to the amount of variance they explain in the original matrix.
    By selecting a subset of these basis vectors (through our choice of dimensionality
    reduction) we can find a lower dimensional space in which to represent the graph.

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012.
    .. [2] Von Luxburg, Ulrike. "A tutorial on spectral clustering," Statistics
        and computing, Vol. 17(4), pp. 395-416, 2007.
    .. [3] Rohe, Karl, Sourav Chatterjee, and Bin Yu. "Spectral clustering and
        the high-dimensional stochastic blockmodel," The Annals of Statistics,
        Vol. 39(4), pp. 1878-1915, 2011.
    """

    def __init__(
        self,
        form: LaplacianFormType = "DAD",
        n_components: Optional[int] = None,
        n_elbows: Optional[int] = 2,
        algorithm: SvdAlgorithmType = "randomized",
        n_iter: int = 5,
        check_lcc: bool = True,
        regularizer: Optional[float] = None,
        concat: bool = False,
        svd_seed: Optional[int] = None,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
            check_lcc=check_lcc,
            concat=concat,
            svd_seed=svd_seed,
        )
        self.form = form
        self.regularizer = regularizer

    def fit(
        self,
        graph: GraphRepresentation,
        y: Optional[Any] = None,
        *args: Any,
        **kwargs: Any
    ) -> "LaplacianSpectralEmbed":
        """
        Fit LSE model to input graph

        By default, uses the Laplacian normalization of the form:

        .. math:: L = D^{-1/2} A D^{-1/2}

        Parameters
        ----------
        graph : array-like, scipy.sparse.csr_matrix, or networkx.Graph
            Input graph to embed. see graspologic.utils.import_graph

        Returns
        -------
        self : object
            Returns an instance of self.
        """
        A = self._fit(graph)
        L_norm = to_laplacian(A, form=self.form, regularizer=self.regularizer)
        self._reduce_dim(L_norm)

        self.is_fitted_ = True

        return self

    def _compute_oos_prediction(self, X, directed):  # type: ignore
        """
        Computes the out-of-sample latent position estimation.
        Parameters
        ----------
        X: np.ndarray
            Input to do oos embedding on.
        directed: bool
            Indication if graph is directed or undirected
        Returns
        -------
        out : array_like or tuple, shape
        """

        if not directed:
            if X.ndim == 1:
                X = np.expand_dims(X, axis=0)

            return ((X @ self._pinv_left).T / np.sum(X, axis=1)).T
        elif directed:
            X_0 = X[0]
            X_1 = X[1]

            if X_0.ndim == 1:
                X_0 = np.expand_dims(X_0, axis=0)
                X_1 = np.expand_dims(X_1, axis=0)

            return ((X_1 @ self._pinv_right).T / np.sum(X_1, axis=1)).T, (
                (X_0 @ self._pinv_left).T / np.sum(X_0, axis=1)
            ).T
