#%%
from graspologic.utils import import_graph, to_laplacian
from graspologic.embed.base import BaseSpectralEmbed
from graspologic.embed.lse import LaplacianSpectralEmbed
from graspologic.simulations import sbm
from graspologic.plot import heatmap

import numpy as np
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import seaborn as sns


class CovariateAssistedEmbedding(BaseSpectralEmbed):
    """
    Perform Spectral Embedding on a graph with covariates, using the regularized graph Laplacian.

    The Covariate-Assisted Spectral Embedding is a k-dimensional Euclidean representation
    of a graph based on its Laplacian (assortative) or squared Laplacian (non-assortative),
    and a vector of covariate features for each node.

    Parameters
    ----------
    n_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
        ``select_dimension`` using ``n_elbows`` argument.

    n_elbows : int, optional, default: 2
        If `n_components=None`, then compute the optimal embedding dimension using
        `select_dimension`. Otherwise, ignored.

    algorithm : {'full', 'truncated' (default), 'randomized'}, optional
        SVD solver to use:
        - 'full'
            Computes full svd using ``scipy.linalg.svd``
        - 'truncated'
            Computes truncated svd using ``scipy.sparse.linalg.svd``
        - 'randomized'
            Computes randomized svd using
            ``sklearn.utils.extmath.randomized_svd``

    n_iter : int, optional (default = 5)
        Number of iterations for randomized SVD solver. Not used by 'full' or
        'truncated'. The default is larger than the default in randomized_svd
        to handle sparse matrices that may have large slowly decaying spectrum.

    check_lcc : bool , optional (defult =True)
        Whether to check if input graph is connected. May result in non-optimal
        results if the graph is unconnected. Not checking for connectedness may
        result in faster computation.

    concat : bool, optional (default = False)
        If graph(s) are directed, whether to concatenate each graph's left and right
        (out and in) latent positions along axis 1.

    assortative : bool, optional (default = False)
        Assortative graphs have a within-group edge probability greater than their
        between-group edge probability; the opposite is true for a non-assortative
        graph.

    References
    ---------
    .. [1] Binkiewicz, N., Vogelstein, J. T., & Rohe, K. (2017). Covariate-assisted
    spectral clustering. Biometrika, 104(2), 361-377.
    """

    def __init__(
        self,
        assortative=False,
        n_components=None,
        n_elbows=2,
        algorithm="randomized",
        n_iter=5,
        check_lcc=True,
        concat=False,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
            check_lcc=check_lcc,
            concat=concat,
        )

        if not isinstance(assortative, bool):
            msg = "Assortativeness of the graph should be a Boolean hyperparameter."
            raise TypeError(msg)
        self.assortative_ = assortative  # TODO: compute this automatically?
        self.is_fitted_ = False

    def fit(self, graph, covariates, y=None):
        """
        Fit a CASE model to an input graph, along with its covariates. If the graph is
        assortative, we embed

        .. math:: L_ = LL + \alpha XX^T

        where :math:`\alpha` is a tuning parameter which makes the leading eigenvectors
        of :math:`LL` and :math:`XX^T` the same. Here, :math:`L` is the regularized
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
        X = covariates.copy()

        # workhorse code
        L = to_laplacian(A, form="R-DAD")
        LL = L if self.assortative_ else L @ L
        XXt = X @ X.T
        a = self._get_tuning_parameter(LL, XXt)
        L_ = LL + a * (XXt)
        self._reduce_dim(L_)

        self.is_fitted_ = True
        return self

    def _get_tuning_parameter(self, LL, XXt):
        """
        Find an alpha which causes the leading eigenvectors of L@L and a*X@X.T to be
        the same.

        Parameters
        ----------
        LL : array
            The squared regularized graph Laplacian
        XXt : array
            X@X.T, where X is the covariate matrix.

        Returns
        -------
        alpha : float
            Tuning parameter which normalizes LL and XXt.
        """
        L_leading = np.linalg.eigvalsh(LL)[-1]
        X_leading = np.linalg.eigvalsh(XXt)[-1]
        return np.float(L_leading / X_leading)
