import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh
from sklearn.preprocessing import normalize, scale

from graspologic.utils import import_graph, to_laplacian, is_almost_symmetric
from graspologic.embed.base import BaseSpectralEmbed


class CovariateAssistedEmbed(BaseSpectralEmbed):
    """
    Perform Spectral Embedding on a graph with covariates, using the regularized graph Laplacian.

    The Covariate-Assisted Spectral Embedding is a k-dimensional Euclidean representation
    of a graph based on a function of its Laplacian and a vector of covariate features
    for each node. For more information, see [1].

    Parameters
    ----------
    alpha : float, optional, default = None
        Tuning parameter to use:
            -  None: Default to the ratio of the leading eigenvector of the Laplacian
                     to the leading eigenvector of the covariate matrix.
            - float: Use a particular alpha-value.

    assortative : bool, default = True
        Embedding algorithm to use. An assortative network is any network where the
        within-group probabilities are greater than the between-group probabilities.
        Here, L is the regularized Laplacian, Y is the covariate matrix, and a is the
        tuning parameter alpha:
            - True: Embed ``L + a*Y@Y.T``. Better for assortative graphs.
            - False: Embed ``L@L + a*Y@Y.T``. Better for non-assortative graphs.

    n_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        n_components must be <= min(Y.shape). Otherwise, n_components must be
        < min(Y.shape). If None, then optimal dimensions will be chosen by
        ``select_dimension`` using ``n_elbows`` argument.

    n_elbows : int, optional, default: 2
        If `n_components=None`, then compute the optimal embedding dimension using
        `select_dimension`. Otherwise, ignored.

    check_lcc : bool , optional, defult = True
        Whether to check if input graph is connected. May result in non-optimal
        results if the graph is unconnected. Not checking for connectedness may
        result in faster computation.


    References
    ---------
    .. [1] Binkiewicz, N., Vogelstein, J. T., & Rohe, K. (2017). Covariate-assisted
    spectral clustering. Biometrika, 104(2), 361-377.
    """

    def __init__(
        self,
        alpha=None,
        assortative=True,
        n_components=None,
        n_elbows=2,
        check_lcc=False,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            check_lcc=check_lcc,
            concat=False,
            algorithm="square",
        )

        if not isinstance(assortative, bool):
            msg = "embedding_alg must be a boolean value."
            raise ValueError(msg)
        self.assortative = assortative

        if not ((alpha is None) or isinstance(alpha, (float, int))):
            msg = "alpha must be in {None, float, int}."
            raise TypeError(msg)

        self.alpha = alpha
        self.latent_right_ = None
        self.is_fitted_ = False

    def fit(self, network, y=None):
        """
        Fit a CASE model to an input graph, along with its covariates. Depending on the
        embedding algorithm, we embed

        .. math:: L_ = LL + \alpha YY^T
        .. math:: L_ = L + \alpha YY^T

        where :math:`\alpha` is a tuning parameter which makes the leading eigenvalues
        of the two summands the same. Here, :math:`L` is the regularized
        graph Laplacian, and :math:`Y` is a matrix of covariates for each node.

        Covariates are row-normalized to unit l2-norm.

        Parameters
        ----------
        network : tuple or list of np.ndarrays
            Contains the tuple (A, Y), where A is an adjacency matrix and Y is the
            matrix of covariates.

            A : array-like or networkx.Graph
                Input graph to embed. See graspologic.utils.import_graph

            Y : array-like, shape (n_vertices, n_covariates)
                Covariate matrix. Each node of the graph is associated with a set of
                `d` covariates. Row `i` of the covariates matrix corresponds to node
                `i`, and the number of columns are the number of covariates.

        y: Ignored

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        # setup
        if not isinstance(network, (tuple, list)):
            msg = "Network should be a tuple-like object of (graph, covariates)."
            raise TypeError(msg)
        if len(network) != 2:
            msg = "Network should be a tuple-like object of (graph, covariates)."
            raise ValueError(msg)

        graph, covariates = network
        A = import_graph(graph)
        n = A.shape[0]
        if n != A.shape[1]:
            raise ValueError("Graph should be square")
        # if not is_almost_symmetric(A):
        #     raise ValueError("Fit an undirected graph")

        # Create regularized Laplacian, scale covariates to unit norm
        L = to_laplacian(A, form="R-DAD")
        Y = covariates
        if Y.ndim == 1:
            Y = Y[:, np.newaxis]
        Y = normalize(Y, axis=0)

        # Use ratio of the two leading eigenvalues
        # if alpha is None
        self._get_tuning_parameter(L, Y)

        # get embedding matrix as a LinearOperator (for computational efficiency)
        mv, rmv = self._matvec(L, Y, a=self.alpha_, assortative=self.assortative)
        L_ = LinearOperator((n, n), matvec=mv, rmatvec=rmv)

        # Dimensionality reduction with SVD
        self._reduce_dim(L_)

        self.is_fitted_ = True
        return self

    def _get_tuning_parameter(self, L, Y):
        """
        Find the alpha which causes the leading eigenspace of LL and YYt to be the same.

        Parameters
        ----------
        L : array
            The regularized graph Laplacian.
        Y : array
            The covariate matrix.

        Returns
        -------
        alpha : float
            Tuning parameter which normalizes the leading eigenspace of LL and YYt.
        """
        # setup
        if self.alpha is not None:
            self.alpha_ = self.alpha
            return self

        # Laplacian leading eigenvector
        (L_top,) = eigsh(L, return_eigenvectors=False, k=1)
        if self.assortative:
            L_top **= 2

        # YY^T leading eigenvector
        n = Y.shape[0]
        YO = LinearOperator((n, n), matvec=lambda v: Y @ (Y.T @ v))
        (YYt_top,) = eigsh(YO, return_eigenvectors=False, k=1)

        # just use the ratio of the leading eigenvalues for the
        # tuning parameter, or the closest value in its possible range.
        self.alpha_ = np.float(L_top / YYt_top)

        return self

    @staticmethod
    def _matvec(L, Y, a=None, assortative=True):
        if assortative:
            mv = lambda v: (L @ v) + a * (Y @ (Y.T @ v))
            rmv = lambda v: (v.T @ L) + a * ((v.T @ Y) @ Y.T)
        else:
            mv = lambda v: (L @ (L @ v)) + a * (Y @ (Y.T @ v))
            rmv = lambda v: (v.T @ L) @ L + a * ((v.T @ Y) @ Y.T)
        return mv, rmv
