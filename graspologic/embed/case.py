from graspologic.utils import import_graph, to_laplacian
from graspologic.embed.base import BaseSpectralEmbed
import numpy as np
import scipy
from joblib import delayed, Parallel
from sklearn.cluster import KMeans
from graspologic.plot import heatmap
from graspologic.utils import remap_labels, is_almost_symmetric
from sklearn.preprocessing import normalize, scale
from sklearn.utils.extmath import randomized_svd
from scipy.sparse.linalg import eigsh


class CovariateAssistedEmbedding(BaseSpectralEmbed):
    """
    Perform Spectral Embedding on a graph with covariates, using the regularized graph Laplacian.

    The Covariate-Assisted Spectral Embedding is a k-dimensional Euclidean representation
    of a graph based on a function of its Laplacian and a vector of covariate features
    for each node. For more information, see [1].

    Parameters
    ----------
    embedding_alg : str, default = "assortative"
        Embedding algorithm to use:
        - "assortative": Embed ``L + a*X@X.T``. Better for assortative graphs.
        - "non-assortative": Embed ``L@L + a*X@X.T``. Better for non-assortative graphs.
        - "cca": Embed ``L@X``. Better for large graphs and faster.

    alpha : float, optional, default = None
        Tuning parameter to use. Not used if embedding_alg == cca:
            -  None: Default to the ratio of the leading eigenvector of the Laplacian
                     to the leading eigenvector of the covariate matrix.
            - float: Use a particular alpha-value.

    n_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
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
        embedding_alg="assortative",
        alpha=None,
        n_components=None,
        n_elbows=2,
        check_lcc=False,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            check_lcc=check_lcc,
            concat=False,
        )

        if embedding_alg not in {"assortative", "non-assortative", "cca"}:
            msg = "embedding_alg must be in {assortative, non-assortative, cca}."
            raise ValueError(msg)
        self.embedding_alg = embedding_alg

        if not ((alpha is None) or isinstance(alpha, float)):
            msg = "alpha must be in {None, float}."
            raise TypeError(msg)

        self.alpha = alpha
        self.latent_right_ = None
        self.is_fitted_ = False

    def fit(self, graph, covariates, y=None, labels=None):
        """
        Fit a CASE model to an input graph, along with its covariates. Depending on the
        embedding algorithm, we embed

        .. math:: L_ = LL + \alpha XX^T
        .. math:: L_ = L + \alpha XX^T
        .. math:: L_ = LX

        where :math:`\alpha` is a tuning parameter which makes the leading eigenvalues
        of the two summands the same. Here, :math:`L` is the regularized
        graph Laplacian, and :math:`X` is a matrix of covariates for each node.

        Parameters
        ----------
        graph : array-like or networkx.Graph
            Input graph to embed. See graspologic.utils.import_graph

        covariates : array-like, shape (n_vertices, n_covariates)
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
        A = import_graph(graph)
        if not is_almost_symmetric(A):
            raise ValueError("Fit an undirected graph")

        # center and scale covariates to unit norm
        covariates = normalize(covariates, axis=0)

        # save necessary params
        L = to_laplacian(A, form="R-DAD")
        X = covariates.copy()

        # change params based on tuning algorithm
        if self.embedding_alg == "cca":
            LL = L @ X
            XXt = 0
        elif self.embedding_alg == "assortative":
            LL = L.copy()
            XXt = X @ X.T
        elif self.embedding_alg == "non-assortative":
            LL = L @ L
            XXt = X @ X.T

        # Get weight and create embedding matrix
        self._get_tuning_parameter(LL, XXt)
        L_ = (LL + self.alpha_ * (XXt)).astype(float)

        # Dimensionality reduction with SVD
        self._reduce_dim(L_)

        self.is_fitted_ = True
        return self

    def _get_tuning_parameter(self, LL, XXt):
        """
        Find the alpha which causes the leading eigenspace of LL and XXt to be the same.

        Parameters
        ----------
        LL : array
            The regularized graph Laplacian (assortative)
            The squared regularized graph Laplacian (non-assortative)
        XXt : array
            X@X.T, where X is the covariate matrix.

        Returns
        -------
        alpha : float
            Tuning parameter which normalizes the leading eigenspace of LL and XXt.
        """
        # setup
        if self.alpha is not None:
            self.alpha_ = self.alpha
            return self
        if self.embedding_alg == "cca":
            self.alpha_ = 0
            return self

        # calculate bounds
        (L_top,) = eigsh(LL, return_eigenvectors=False, k=1)
        (X_top,) = eigsh(XXt, return_eigenvectors=False, k=1)

        # just use the ratio of the leading eigenvalues for the
        # tuning parameter, or the closest value in its possible range.
        self.alpha_ = np.float(L_top / X_top)

        return self
