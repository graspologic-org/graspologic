# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
from .base import BaseEmbed
from ..utils import import_graph, to_laplace, is_fully_connected
from .svd import selectSVD
import numpy as np
import scipy.sparse as sp
from sklearn.cluster import KMeans


class CovariateAssistedSpectralEmbed(BaseEmbed):
    r"""
    Class performs Covaariates Assisted Spectual Embedding and Clustering
        
    Class for computing the k-means clustering results of a graph with the 
    Laplacian Matrix and Covariates Matrix generated from regression.[1]_.It 
    use the a weighted sum of Laplacian Matrix and Covariates :
        
    .. math:: L'=L+h*XX^T 
    or 
    
    .. math:: L'=L^2+h*XX^T
    depending on if the graph is assortative.It relies on an 
    optimized implicitly restarted Lanczos bidiagonalization SVD decomposition 
    to cluster the
    nodes of the graph in to k clusters[1]_[2]_.
    
    

    Read more in the :ref:`tutorials <embed_tutorials>`

    Parameters
    ----------
    form : {'DAD' (default), 'I-DAD', 'R-DAD'}, optional
        Specifies the type of Laplacian normalization to use.

    n_components : int or None, default = None
        Desired numbers of clusters.

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
        ``form`` == 'R-DAD'.
    row_norm : bool , optional (default=False)
        if True .The clastering process will be based on row-normlized singular
        vectors.
    cca : bool , optional (default=False)
        If True uses cca method. Recommended for large number of clusters.see[1]_.
    assortative : bool , optional (default=True)
        If True uses assortative CASC method. For SBM models assortative means 
        within-block probability is higher than trans-block probability.see[1]_.
    
    Attributes
    ----------
    latent_left_ : array, shape (n_samples, n_components)
        Estimated left latent positions of the graph.
    labels_ :  array, shape (n_samples)
        Estimated clusters of the graph nodes.
    opt_h_ : double
        optimized tuning parameter.
    inertia_ : double
        kmeans_results.inertia_ of the Kmeans clustering
  

    See Also
    --------
    graspy.embed.selectSVD
    graspy.embed.select_dimension
    graspy.utils.to_laplace
    sklearn.cluster.KMeans
    

    References
    ----------
    .. [1] N. Binkiewicz, J. T. Vogelstein, K. Rohe, Covariate-assisted spectral
       clustering, Biometrika, Volume 104, Issue 2, June 2017, Pages 361–377
       
    .. [2] Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
       J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005
    """

    def __init__(
        self,
        cca=False,
        form="R-DAD",
        n_components=None,
        n_elbows=2,
        algorithm="randomized",
        n_iter=5,
        check_lcc=True,
        regularizer=1,
        assortative=True,
        row_norm=False,
        n_points=100,
    ):
        super().__init__(
            n_components=n_components,
            n_elbows=n_elbows,
            algorithm=algorithm,
            n_iter=n_iter,
            check_lcc=check_lcc,
        )
        self.form = form
        self.regularizer = regularizer
        self.assortative = assortative
        self.row_norm = row_norm
        self.n_points = n_points
        self.cca = cca
        self.labels_ = None
        self.opt_h_ = None
        self.inertia_ = None

    #    @timethis
    def fit(self, graph, covariate_matrix, y=None):
        """
        Fit CASC model to input graph

        

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to embed. see graspy.utils.import_graph
        covariate:array_like ,shape(n_verts,n_covariates)
            Bernoulli Covariate Matrix of a graph. 

        y : Ignored

        Returns
        -------
        self : returns an instance of self.
        
        """
        A = import_graph(graph)

        if self.check_lcc:
            if not is_fully_connected(A):
                msg = (
                    "Input graph is not fully connected. Results may not"
                    + "be optimal. You can compute the largest connected component by"
                    + "using ``graspy.utils.get_lcc``."
                )
                warnings.warn(msg, UserWarning)

        l_morn = to_laplace(A, form=self.form, regularizer=self.regularizer)

        [hmax, hmin] = self.get_tuning_range(
            l_morn, covariate_matrix, self.n_components, self.assortative, self.cca
        )

        res = self.get_casc_clusters(
            l_morn,
            covariate_matrix,
            hmin,
            hmax,
            self.n_components,
            self.n_points,
            self.row_norm,
            self.assortative,
            self.cca,
        )
        self.latent_left_ = res["casc_svd"]
        self.labels_ = res["cluster"]
        self.opt_h_ = res["h"]
        self.inertia_ = res["wcss"]
        return self

    def fit_transform(self, graph, covariate_matrix, y=None):
        """
        Fit the CASC model with graphs and apply the transformation.

        

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to embed. see graspy.utils.import_graph
            
        covariate:array_like ,shape(n_verts,n_covariates)
            Bernoulli Covariate Matrix of a graph. 

        y : Ignored
        

        Returns
        -------
        out : np.ndarray, shape (n_vertices, n_dimension)
            A single np.ndarray represents the latent position of an undirected
            graph
        """
        self.fit(graph, covariate_matrix, y=None)
        return self.latent_left_

    def fit_predict(self, graph, covariate_matrix, y=None, return_full=True):
        """
        Fit the CASC model with graphs and predict clusters.

        

        Parameters
        ----------
        graph : array_like or networkx.Graph
            Input graph to embed. see graspy.utils.import_graph
            
        covariate:array_like ,shape(n_verts,n_covariates)
            Bernoulli Covariate Matrix of a graph. 

        y : Ignored
        
        return_full : bool , optional (default=True)
            If True, returns the detailed clustering information including 
            optimized turning parameter, embedding results and inertia. Else
            returns labels only.

        Returns
        -------
        labels_ : np.ndarray, shape (n_vertices)
            Component labels of vertices.
            
        latent_left_ : np.ndarray, shape (n_vertices, n_dimension)
            A single np.ndarray represents the latent position of an undirected
            graph
            
        opt_h_ : double
            optimized tuning parameter.
            
        inertia_ : double
            kmeans_results.inertia_ of the Kmeans clustering
            
        """

        self.fit(graph, covariate_matrix, y=None)
        if return_full:
            return self.labels_, self.latent_left_, self.opt_h_, self.inertia_
        else:
            return self.labels_

    def get_tuning_range(self, graph_matrix, covariates, n_blocks, assortative, cca):
        n_cov = covariates.shape[1]

        U, D, V = selectSVD(
            covariates,
            n_components=covariates.shape[1],
            n_elbows=self.n_elbows,
            algorithm="full",
        )

        min_tmp = np.min([n_cov, n_blocks])
        sing_val_cov = D[0:min_tmp]

        if assortative:
            u1, d1, v1 = selectSVD(
                graph_matrix,
                n_components=n_blocks + 1,
                n_elbows=self.n_elbows,
                algorithm="randomized",
            )
            tmp1 = n_blocks + 1
            eigen_val_graph = d1[0:tmp1]
            if n_cov > n_blocks:
                hmax = eigen_val_graph[0] / (
                    sing_val_cov[n_blocks - 1] ** 2 - sing_val_cov[n_blocks] ** 2
                )
            else:
                hmax = eigen_val_graph[0] / sing_val_cov[n_cov - 1] ** 2
            hmin = (
                eigen_val_graph[n_blocks - 1] - eigen_val_graph[n_blocks]
            ) / sing_val_cov[0] ** 2
        else:
            u1, d1, v1 = selectSVD(
                graph_matrix,
                n_components=n_blocks + 1,
                n_elbows=self.n_elbows,
                algorithm="randomized",
            )
            tmp1 = n_blocks + 1
            eigen_val_graph = d1[0:tmp1] ** 2
            if n_cov > n_blocks:
                hmax = eigen_val_graph[0] / (
                    sing_val_cov[n_blocks - 1] ** 2 - sing_val_cov[n_blocks] ** 2
                )
            else:
                hmax = eigen_val_graph[0] / sing_val_cov[n_cov - 1] ** 2
            hmin = (
                eigen_val_graph[n_blocks - 1] - eigen_val_graph[n_blocks]
            ) / sing_val_cov[0] ** 2
        return [hmax, hmin]

    def get_casc_clusters(
        self,
        graph_mat,
        covariates,
        hmin,
        hmax,
        n_blocks,
        n_points,
        row_norm,
        assortative,
        cca,
    ):

        h_tuning_seq = np.linspace(hmax, hmin, n_points)
        wcss_vec = []

        orthoX = []
        orthoL = []
        if not cca:
            for i in range(n_points):
                casc_results = self.get_casc_results(
                    graph_mat,
                    covariates,
                    h_tuning_seq[i],
                    n_blocks,
                    row_norm,
                    assortative,
                    cca,
                )
                orthoL.append(casc_results["orthoL"])
                orthoX.append(casc_results["orthoX"])
                wcss_vec.append(casc_results["wcss"])

            h_opt = h_tuning_seq[wcss_vec.index(min(wcss_vec))]
        else:
            h_opt = 0
        h_opt_results = self.get_casc_results(
            graph_mat, covariates, h_opt, n_blocks, row_norm, assortative, cca
        )
        return {
            "cluster": h_opt_results["cluster"],
            "h": h_opt,
            "wcss": h_opt_results["wcss"],
            "casc_svd": h_opt_results["casc_svd"],
        }

    def get_ortho(
        self, graph_mat, covariates, casc_svd_evec, casc_svd_eval, h, n_blocks
    ):
        orthoL = (
            casc_svd_evec[:, (n_blocks - 1)]
            .transpose()
            .dot(graph_mat)
            .dot(casc_svd_evec[:, (n_blocks - 1)])
            .transpose()
            / casc_svd_eval[(n_blocks - 1)]
        )
        orthoX = h * (
            casc_svd_evec[:, (n_blocks - 1)]
            .transpose()
            .dot(covariates)
            .dot(covariates.transpose())
            .dot(casc_svd_evec[:, (n_blocks - 1)])
            / casc_svd_eval[(n_blocks - 1)]
        )
        return [orthoL / (orthoL + orthoX), orthoX / (orthoL + orthoX)]

    def get_casc_svd(
        self, graph_mat, covariates, h_tuning_param, n_blocks, assortative, cca
    ):
        if cca:
            new_laplacian = np.dot(graph_mat, covariates)
            u2, d2, v2 = selectSVD(
                new_laplacian,
                n_components=min(new_laplacian.shape),
                n_elbows=self.n_elbows,
                algorithm="full",
            )

        else:
            if assortative:

                def matmult(x, y, t=False):
                    if t:
                        res = np.dot(
                            graph_mat.transpose(), y
                        ) + h_tuning_param * np.dot(
                            covariates, np.dot(covariates.transpose(), y)
                        )
                    else:
                        res = np.dot(graph_mat, y) + h_tuning_param * np.dot(
                            covariates, np.dot(covariates.transpose(), y)
                        )
                    return res

            else:

                def matmult(x, y, t=False):
                    if t:
                        res = np.dot(
                            graph_mat, np.dot(graph_mat.transpose(), y)
                        ) + h_tuning_param * np.dot(
                            covariates, np.dot(covariates.transpose(), y)
                        )
                    else:
                        res = np.dot(
                            graph_mat, np.dot(graph_mat, y)
                        ) + h_tuning_param * np.dot(
                            covariates, np.dot(covariates.transpose(), y)
                        )
                    return res

            u2, d2, v2, it, pr = self._irlb(graph_mat, n_blocks + 1, matmult)

        eVec = u2[:, 0:n_blocks]
        eVal = d2[0 : (n_blocks + 1)]

        return {"eVec": eVec, "eVal": eVal}

    def get_casc_results(
        self,
        graph_mat,
        covariates,
        h_tuning_param,
        n_blocks,
        row_norm,
        assortative,
        cca,
    ):
        casc_svd = self.get_casc_svd(
            graph_mat, covariates, h_tuning_param, n_blocks, assortative, cca
        )

        if row_norm:
            casc_svd_tmp = casc_svd["eVec"] / np.sqrt(sum(casc_svd["eVec"] ** 2))
        else:
            casc_svd_tmp = casc_svd["eVec"]
        kmeans_results = KMeans(n_clusters=n_blocks).fit(casc_svd_tmp)

        if not cca:
            ortho = self.get_ortho(
                graph_mat,
                covariates,
                casc_svd["eVec"],
                casc_svd["eVal"],
                h_tuning_param,
                n_blocks,
            )

            return {
                "orthoL": ortho[0],
                "orthoX": ortho[1],
                "wcss": kmeans_results.inertia_,
                "cluster": kmeans_results.labels_,
                "casc_svd": casc_svd_tmp,
            }

        else:
            return {
                "orthoL": None,
                "orthoX": None,
                "wcss": kmeans_results.inertia_,
                "cluster": kmeans_results.labels_,
                "casc_svd": casc_svd_tmp,
            }

    def mult(self, A, x, t=False):
        assert x.ndim == 1
        if sp.issparse(A):
            if t:
                return sp.csr_matrix(x).dot(A).transpose().todense().A[:, 0]
            return A.dot(sp.csr_matrix(x).transpose()).todense().A[:, 0]
        if t:
            return np.asarray(A.transpose().dot(x)).ravel()
        return np.asarray(A.dot(x)).ravel()

    def _orthog(self, Y, X):
        """Orthogonalize a vector or matrix Y against the columns of the matrix X.
        This function requires that the column dimension of Y is less than X and
        that Y and X have the same number of rows.
        """
        dotY = Y.dot(X)
        return Y - self.mult(X, dotY)

    # Simple utility function used to check linear dependencies during computation:
    def invcheck(self, x):
        eps2 = 2 * np.finfo(np.float).eps
        if x > eps2:
            return 1.0 / x
        else:
            warnings.warn("Ill-conditioning encountered, result accuracy may be poor")
            return 0.0

    def _irlb(
        self,
        A,
        n,
        matmult,
        tol=0.0001,
        maxit=50,
        center=None,
        scale=None,
        random_state=0,
    ):
        """
        Estimate a few of the largest singular values and corresponding singular
        vectors of matrix using the implicitly restarted Lanczos bidiagonalization
        method of Baglama and Reichel.
        The algorithm estimates the truncated singular value decomposition:
        .. math::A.dot(X[2]) = X[0]*X[1].
        
    
        Parameters
        ----------
        A : Array-like
            Input Matrix for SVD Decomposition.
        n : int
            Desides number of sigular value and singular vector.
        matmult :function pointer 
            A function do the multiply in the SVD decomposition for performance
            optimization.
            
        tol : float , optional(default=0.0001)
            An estimation tolerance. Smaller means more accurate estimates.
        maxit : int , optional(default=50)
            Maximum number of Lanczos iterations allowed.
        

        

        Returns
        -------
        X[0]:A j * nu matrix of estimated left singular vectors.
        X[1]:A vector of length nu of estimated singular values.
        X[2]:A k * nu matrix of estimated right singular vectors.
        X[3]:The number of Lanczos iterations run.
        X[4]:The number of matrix-vector products run.

        Reference
        -------
        1.Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
        J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005
        
        """
        np.random.seed(random_state)

        # Numpy routines do undesirable things if these come in as N x 1 matrices instead of size N arrays
        if center is not None and not isinstance(center, np.ndarray):
            raise TypeError("center must be a numpy.ndarray")
        if scale is not None and not isinstance(scale, np.ndarray):
            raise TypeError("scale must be a numpy.ndarray")

        nu = n
        m = A.shape[0]
        n = A.shape[1]
        if min(m, n) < 2:
            raise Exception("The input matrix must be at least 2x2.")
        # TODO: More efficient to have a path that performs a standard SVD
        # if over half the eigenvectors are requested
        m_b = min((nu + 20, 3 * nu, min(A.shape)))  # Working dimension size
        # m_b = nu + 7 # uncomment this line to check for similar results with R package
        mprod = 0
        it = 0
        j = 0
        k = nu
        smax = 1

        V = np.zeros((n, m_b))  # Approximate right vectors
        W = np.zeros((m, m_b))  # Approximate left vectors
        F = np.zeros((n, 1))  # Residual vector
        B = np.zeros((m_b, m_b))  # Bidiagonal approximation

        V[:, 0] = np.random.randn(n)  # Initial vector
        V[:, 0] = V[:, 0] / np.linalg.norm(V)

        while it < maxit:
            if it > 0:
                j = k

            VJ = V[:, j]

            # apply scaling
            if scale is not None:
                VJ = VJ / scale

            W[:, j] = matmult(A, VJ)
            mprod += 1

            # apply centering

            if center is not None:
                W[:, j] = W[:, j] - np.dot(center, VJ)

            if it > 0:
                # NB W[:,0:j] selects columns 0,1,...,j-1
                W[:, j] = self._orthog(W[:, j], W[:, 0:j])
            s = np.linalg.norm(W[:, j])
            sinv = self.invcheck(s)
            W[:, j] = sinv * W[:, j]

            # Lanczos process
            while j < m_b:
                F = matmult(A, W[:, j], t=True)
                #
                mprod += 1

                # apply scaling
                if scale is not None:
                    F = F / scale
                # apply centering, note for cases where center is the column
                # mean, this correction is often equivalent to a no-op as
                # np.sum(W[:, j]) is often close to zero
                if center is not None:
                    F = F - np.sum(W[:, j]) * center
                F = F - s * V[:, j]
                F = self._orthog(F, V[:, 0 : (j + 1)])
                fn = np.linalg.norm(F)
                fninv = self.invcheck(fn)
                F = fninv * F
                if j < m_b - 1:
                    V[:, j + 1] = F
                    B[j, j] = s
                    B[j, j + 1] = fn
                    VJp1 = V[:, j + 1]

                    # apply scaling
                    if scale is not None:
                        VJp1 = VJp1 / scale

                    W[:, j + 1] = matmult(A, VJp1)
                    #
                    mprod += 1

                    # apply centering

                    if center is not None:
                        W[:, j + 1] = W[:, j + 1] - np.dot(center, VJp1)

                    # One step of classical Gram-Schmidt...
                    W[:, j + 1] = W[:, j + 1] - fn * W[:, j]
                    # ...with full reorthogonalization
                    W[:, j + 1] = self._orthog(W[:, j + 1], W[:, 0 : (j + 1)])
                    s = np.linalg.norm(W[:, j + 1])
                    sinv = self.invcheck(s)
                    W[:, j + 1] = sinv * W[:, j + 1]
                else:
                    B[j, j] = s
                j += 1
            # End of Lanczos process
            S = np.linalg.svd(B)
            R = fn * S[0][m_b - 1, :]  # Residuals
            if it < 1:
                smax = S[1][0]  # Largest Ritz value
            else:
                smax = max((S[1][0], smax))

            conv = sum(np.abs(R[0:nu]) < tol * smax)
            if conv < nu:  # Not coverged yet
                k = max(conv + nu, k)
                k = min(k, m_b - 3)
            else:
                break
            # Update the Ritz vectors
            V[:, 0:k] = V[:, 0:m_b].dot(S[2].transpose()[:, 0:k])
            V[:, k] = F
            B = np.zeros((m_b, m_b))
            # Improve this! There must be better way to assign diagonal...
            for l in range(0, k):
                B[l, l] = S[1][l]
            B[0:k, k] = R[0:k]
            # Update the left approximate singular vectors
            W[:, 0:k] = W[:, 0:m_b].dot(S[0][:, 0:k])
            it += 1

        U = W[:, 0:m_b].dot(S[0][:, 0:nu])
        V = V[:, 0:m_b].dot(S[2].transpose()[:, 0:nu])
        return (U, S[1][0:nu], V, it, mprod)
