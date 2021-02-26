# %%
from graspologic.utils import import_graph, to_laplacian
from graspologic.embed.base import BaseSpectralEmbed
from graspologic.embed.lse import LaplacianSpectralEmbed
from graspologic.simulations import sbm
from graspologic.plot import heatmap

import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns


class CovariateAssistedEmbedding(BaseSpectralEmbed):
    """
    Perform Spectral Embedding on a graph with covariates, using the regularized graph Laplacian.

    The Covariate-Assisted Spectral Embedding is a k-dimensional Euclidean representation
    of a graph based on a function of its Laplacian and a vector of covariate features
    for each node.

    Parameters
    ----------
    embedding_alg : str, default = "assortative"
        Embedding algorithm to use:
        - "assortative": Embed ``L + a*X@X.T``. Better for assortative graphs.
        - "non-assortative": Embed ``L@L + a*X@X.T``. Better for non-assortative graphs.
        - "cca": Embed ``L@X``. Better for large graphs.

    n_components : int or None, default = None
        Desired dimensionality of output data. If "full",
        n_components must be <= min(X.shape). Otherwise, n_components must be
        < min(X.shape). If None, then optimal dimensions will be chosen by
        ``select_dimension`` using ``n_elbows`` argument.

    alpha : float, optional (default = None)
        Tuning parameter to use. Not used if embedding_alg == cca:
            -  None: tune the alpha-value by minimizing the k-means objective function. 
                     Since this involves running k-means over a parameter space, this 
                     will result in a slower algorithm in exchange for likely better 
                     clustering.
            - float: use a particular alpha-value. Results in a much faster algorithm 
                     (since we are not tuning with kmeans) in exchange for potentially
                     suboptimal clustering results.
            -    -1: Default to the ratio of the leading eigenvector of the Laplacian
                     to the leading eigenvector of the covariate matrix. This will
                     result in suboptimal clustering in exchange for increased
                     clustering speed.

    n_iter : int, optional (default = 100)
        If tuning alpha with k-means, this parameter determines the number of times
        k-means is run. Higher values are more computationally expensive in exchange
        for a finer-grained search of the parameter space.

    n_elbows : int, optional, default: 2
        If `n_components=None`, then compute the optimal embedding dimension using
        `select_dimension`. Otherwise, ignored.

    check_lcc : bool , optional (defult =True)
        Whether to check if input graph is connected. May result in non-optimal
        results if the graph is unconnected. Not checking for connectedness may
        result in faster computation.

    concat : bool, optional (default = False)
        If graph(s) are directed, whether to concatenate each graph's left and right
        (out and in) latent positions along axis 1.


    References
    ---------
    .. [1] Binkiewicz, N., Vogelstein, J. T., & Rohe, K. (2017). Covariate-assisted
    spectral clustering. Biometrika, 104(2), 361-377.
    """

    def __init__(
        self,
        embedding_alg="assortative",
        alpha=None,
        n_iter=100,
        n_components=None,
        n_elbows=2,
        check_lcc=False,
        concat=False
    ):
        super().__init__(
            algorithm="full",
            n_components=n_components,
            n_elbows=n_elbows,
            check_lcc=check_lcc,
            concat=concat
        )

        if embedding_alg not in {"assortative", "non-assortative", "cca"}:
            msg = "embedding_alg must be in {assortative, non-assortative, cca}."
            raise ValueError(msg)
        self.embedding_alg = embedding_alg  # TODO: compute this automatically?

        if not ((alpha is None) or alpha == -1 or isinstance(alpha, float)):
            msg = "alpha must be in {None, float, -1} and must be positive."
            raise TypeError(msg)
        self.alpha = alpha

        self.n_iter = n_iter
        self.is_fitted_ = False

    def fit(self, graph, covariates, y=None):
        """
        Fit a CASE model to an input graph, along with its covariates. Depending on the
        embedding algorithm, we embed

        .. math:: L_ = LL + \alpha XX^T
        .. math:: L_ = L + \alpha XX^T
        .. math:: L_ = LX

        where :math:`\alpha` is a tuning parameter which makes the leading eigenvectors
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

        # save necessary params
        self._L = to_laplacian(A, form="R-DAD")
        self._R = np.shape(covariates)[1]
        self._X = covariates.copy()

        # change params based on tuning algorithm
        if self.embedding_alg == "cca":
            self._LL = self._L@self._X
            self._XXt = self._X
            self.alpha_ = 0
        if self.embedding_alg == "assortative":
            self._LL = self._L
            self._XXt = self._X@self._X.T
            self.alpha_ = self._get_tuning_parameter()
        elif self.embedding_alg == "non-assortative":
            self._LL = self._L@self._L
            self._XXt = self._X@self._X.T
            self.alpha_ = self._get_tuning_parameter()

        self._embed()
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return self._fit_transform(fit=False)
        
    def _get_tuning_parameter(self):
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
        # setup
        if isinstance(self.alpha, (int, float)) and self.alpha != -1:
            return self.alpha
        K = self.n_components  # number of clusters
        R = self._R  # number of covariates
        I = int(R <= K)
        LL = self._LL
        XXt = self._XXt

        # add a 0 to eigvals for 1-indexing (easier to deal with), sort descending
        L_eigvals = np.concatenate((0, np.flip(np.linalg.eigvalsh(LL))), axis=None)
        X_eigvals = np.concatenate((0, np.flip(np.linalg.eigvalsh(XXt))), axis=None)

        # calculate bounds
        L_top = L_eigvals[1]
        X_top = X_eigvals[1]
        amin = (L_eigvals[K] - L_eigvals[K+1])/X_top
        amax = L_top / ((X_eigvals[R] * I) + ((X_eigvals[K] - X_eigvals[K+1])*(1-I)))

        if self.alpha == -1:
            # just use the ratio of the leading eigenvalues for the 
            # tuning parameter, or the closest value in its possible range.
            alpha = np.float(L_top / X_top)
            if amin <= alpha <= amax:
                return alpha
            elif alpha < amin:
                return amin
            elif alpha > amax:
                return amax

        # run kmeans clustering and set alpha to the value
        # which minimizes clustering intertia
        # TODO: optimize... maybe with sklearn.metrics.make_scorer
        #       and a GridSearch? 
        alpha_range = np.linspace(amin, amax, num=self.n_iter)
        inertias = {}
        for a in alpha_range:
            kmeans = KMeans(n_clusters=self.n_components, n_jobs=-1)
            X_ = self._embed(alpha=a)
            kmeans.fit(X_)
            print(f"inertia at {a}: {kmeans.inertia_}")
            inertias[a] = kmeans.inertia_
        alpha = min(inertias, key=inertias.get)
        return alpha


    def _embed(self, alpha=None):
        if alpha is None:
            alpha = self.alpha_

        L_ = self._LL + alpha * (self._XXt)
        self._reduce_dim(L_)
        return self.latent_left_


# def gen_covariates(labels, m1, m2, agreement=1, d=3):
#     """
#     n x 3 matrix of covariates

#     """
#     N = len(labels)

#     # Generate dynamic (e.g., random) covariate array
#     dynamic = np.random.choice([1, 0], p=[m2, 1-m2], size=(N, d))
#     m1_array = np.random.choice([1, 0], p=[m1, 1-m1], size=N)
#     dynamic[np.arange(N), labels] = m1_array
#     return dynamic

# from graspologic.simulations import sbm
# from graspologic.plot import heatmap, pairplot
# A, labels = sbm([100, 100, 100], p=[[.1, .9, .9], [.9, .1, .9], [.9, .9, .1]], return_labels=True)
# X = gen_covariates(labels, m1=.8, m2=.2) 
# case = CovariateAssistedEmbedding(n_components=3, embedding_alg="non-assortative")
# Xhat = case.fit_transform(A, covariates=X)
# pairplot(Xhat, labels=labels)