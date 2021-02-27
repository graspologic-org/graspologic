# %%
from graspologic.utils import import_graph, to_laplacian
from graspologic.embed.base import BaseSpectralEmbed
from graspologic.embed.svd import selectSVD
import numpy as np
import scipy
from sklearn.cluster import KMeans
from graspologic.plot import heatmap
from graspologic.utils import remap_labels

np.set_printoptions(suppress=True)

#%%


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
            - "cca": Embed ``L@X``. Better for large graphs and faster.
    `
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
        concat=False,
    ):
        super().__init__(
            algorithm="full",
            n_components=n_components,
            n_elbows=n_elbows,
            check_lcc=check_lcc,
            concat=concat,
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
        self.latent_right_ = None  # doesn't work for directed graphs atm
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

        # save necessary params
        self._L = to_laplacian(A, form="R-DAD")
        self._R = np.shape(covariates)[1]
        self._X = covariates.copy()

        # change params based on tuning algorithm
        if self.embedding_alg == "cca":
            self._LL = self._L @ self._X
            self._XXt = self._X
            self.alpha_ = 0
        elif self.embedding_alg == "assortative":
            self._LL = self._L
            self._XXt = self._X @ self._X.T
            self.alpha_ = self._get_tuning_parameter()
        elif self.embedding_alg == "non-assortative":
            self._LL = self._L @ self._L
            self._XXt = self._X @ self._X.T
            self.alpha_ = self._get_tuning_parameter()

        # self._embed(plot=True)
        L_ = self._LL + self.alpha_ * (self._XXt)
        self._reduce_dim(L_)
        self.is_fitted_ = True
        # # FOR DEBUGGING
        # kmeans = KMeans(n_clusters=3)
        # labels_ = kmeans.fit_predict(self.latent_left_)
        # labels_ = remap_labels(labels, labels_)
        # print(f"misclustering: {np.count_nonzero(labels - labels_) / len(labels)}")

        # FOR DEBUGGING
        return self

    def _get_tuning_parameter(self):
        """
        Find an alpha within a range which optimizes the k-means objective function on
        our embedding.

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
        n_clusters = self.n_components  # number of clusters
        n_cov = self._R  # number of covariates
        I = int(n_cov <= n_clusters)
        LL = self._LL
        XXt = self._XXt

        # grab eigenvalues
        _, D, _ = selectSVD(self._X, n_components=self._X.shape[1], algorithm="full")
        X_eigvals = D[0 : np.min([n_cov, n_clusters])]
        _, D, _ = selectSVD(self._L, n_components=n_clusters + 1)
        L_eigvals = D[0 : n_clusters + 1]
        if self.embedding_alg == "non-assortative":
            L_eigvals = L_eigvals ** 2

        # calculate bounds
        L_top = L_eigvals[0]
        X_top = X_eigvals[0]
        amin = (L_eigvals[n_clusters - 1] - L_eigvals[n_clusters]) / X_top ** 2
        if n_cov > n_clusters:
            amax = L_top / (X_eigvals[n_clusters - 1] ** 2 - X_eigvals[n_clusters] ** 2)
        else:
            amax = L_top / X_eigvals[n_cov - 1] ** 2

        print(f"{amin=:.9f}, {amax=:.9f}")
        print(f"alpha without tuning: {np.float(L_top / X_top)}")

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
            kmeans = KMeans(n_clusters=n_clusters, n_jobs=-1)
            X_hat = self._embed(alpha=a)
            kmeans.fit(X_hat)
            print(f"inertia at {a:.5f}: {kmeans.inertia_:.5f}")
            inertias[a] = kmeans.inertia_
        alpha = min(inertias, key=inertias.get)
        print(f"Best inertia at alpha={alpha:5f}: {inertias[alpha]:5f}")

        # FOR DEBUGGING
        kmeans = KMeans(n_clusters=self.n_components, n_jobs=-1)
        X_ = self._embed(alpha=np.float(L_top / X_top))
        kmeans.fit(X_)
        print(f"inertia at default alpha {a:.5f}: {kmeans.inertia_:.5f}")

        return alpha

    def _embed(self, alpha=None, plot=False):
        if alpha is None:
            alpha = self.alpha_

        L_ = self._LL + alpha * (self._XXt)
        if plot:
            import matplotlib.pyplot as plt

            # sns.heatmap(self._X, ax=axs[0, 0])
            # heatmap(self._XXt)
            # heatmap(self._L)
            # heatmap(self._LL)
            heatmap(L_)

        U, D, V = selectSVD(
            L_,
            n_components=self.n_components,
            n_elbows=self.n_elbows,
            algorithm="full",
            n_iter=5,
        )

        return U @ np.diag(np.sqrt(D))


# def gen_covariates(labels, m1=0.8, m2=0.2, agreement=1, d=3):
#     """
#     n x 3 matrix of covariates

#     """
#     N = len(labels)
#     d = 3
#     B = np.full((d, d), m2)
#     B[np.diag_indices_from(B)] = m1
#     base = np.eye(d)
#     membership = np.zeros((N, d))
#     n_misassign = 0
#     for i in range(0, N):
#         assign = bool(np.random.binomial(1, agreement))
#         if assign:
#             membership[i, :] = base[labels[i], :]
#         else:
#             membership[i, :] = base[(labels[i] + 1) % (max(labels) + 1), :]
#             n_misassign += 1

#     probs = membership @ B

#     covariates = np.zeros(probs.shape)
#     for i in range(N):
#         for j in range(d):
#             covariates[i, j] = np.random.binomial(1, probs[i, j])

#     return covariates


# def get_misclustering(A, model, labels, covariates=None) -> float:
#     if covariates is None:
#         Xhat = model.fit_transform(A)
#     else:
#         Xhat = model.fit_transform(A, covariates=covariates)

#     kmeans = KMeans(n_clusters=3)
#     labels_ = kmeans.fit_predict(Xhat)

#     # to account for nonidentifiability
#     labels_ = remap_labels(labels, labels_)
#     misclustering = np.count_nonzero(labels - labels_) / len(labels)

#     return misclustering


# from graspologic.simulations import sbm
# from graspologic.utils import remap_labels
# from graspologic.plot import pairplot

# n = 500
# assortative = False
# p, q = 0.03, 0.015
# if not assortative:
#     p, q = q, p
# A, labels = sbm(
#     [n, n, n],
#     p=[[p, q, q], [q, p, q], [q, q, p]],
#     return_labels=True,
# )
# X = gen_covariates(labels, m1=0.8, m2=0.2, agreement=0.0)
# X = gen_covariates(labels, m1=0.8, m2=0.2, agreement=1)
# import seaborn as sns


# case = CovariateAssistedEmbedding(n_components=3, embedding_alg="cca", n_iter=100)
# Xhat = case.fit_transform(A, covariates=X, labels=labels)
# pairplot(Xhat, labels=labels)
# %%
