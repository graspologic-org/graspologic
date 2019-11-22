# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 19:46:01 2019

@author: jerryyao
"""

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
    Class for computing the k-means clustering results of a graph with the 
    Laplacian Matrix and Covariates Matrix generated from regression.[1]_.It 
    use the a weighted sum of Laplacian Matrix and Covariates L'=L+h*X(XT) or 
    L'=L^2+h*X(XT) depending on if the graph is assortative.It relies on an 
    optimized irlb SVD to cluster the nodes of the graph in to k clusters[1][2]_.
    
    

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
    rownorm : bool , optional (default=False)
        if True .The clastering process will be based on row-normlized singular
        vectors.
    CCA : bool , optional (default=False)
        If True uses CCA method. Recommended for large number of clusters.see[1]_.
    Assortative : bool , optional (default=True)
        If True uses Assortative CASC method. For SBM models Assortative means 
        within-block probability is higher than trans-block probability.see[1]_.
    
  

    See Also
    --------
    graspy.embed.selectSVD
    graspy.embed.select_dimension
    graspy.utils.to_laplace
    https://github.com/bwlewis/irlbpy
    sklearn.cluster.KMeans
    

    References
    ----------
        [1]N. Binkiewicz, J. T. Vogelstein, K. Rohe, Covariate-assisted spectral
        clustering, Biometrika, Volume 104, Issue 2, June 2017, Pages 361–377, 
        https://doi.org/10.1093/biomet/asx008
        [2]Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
        J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005
    """

    def __init__(
        self,
        CCA=False,
        form="R-DAD",
        n_components=None,
        n_elbows=2,
        algorithm="randomized",
        n_iter=5,
        check_lcc=True,
        regularizer=1,
        assortative=True,
       
        row_norm=False,
        n_points=100
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
        self.row_norm=row_norm
        self.n_points=n_points
        self.CCA=CCA
        

    
#    @timethis
    def fit(self, graph, covariate_matrix,y=None):
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
        Returns a dict containing following keywords:
        cluster : Clustering labels
        
        h : optimized tuning parameter
        
        wcss : kmeansResults.inertia_ of the Kmeans clustering
        
        cascSvd : Svd dict containing the Svd decomposation resluts. 
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

        L_norm = to_laplace(A, form=self.form, regularizer=self.regularizer)
        
        [hmax,hmin]=self.get_tuning_range(L_norm,covariate_matrix,
        self.n_components,self.assortative,self.CCA)
        
        res=self.getCascClusters(L_norm, covariate_matrix,hmin,hmax, 
        self.n_components,self.n_points, self.row_norm, self.assortative,self.CCA)
        
        return res
#    @timethis
    def get_tuning_range(self,graphMatrix, covariates, nBlocks,assortative,CCA):
        nCov = covariates.shape[1]
        
        U, D, V = selectSVD(covariates,n_components=covariates.shape[1],
                            n_elbows=self.n_elbows,algorithm='full')    
        
        min_tmp = np.min([nCov,nBlocks])
        singValCov = D[0:min_tmp]
        
        if assortative:
            u1,d1,v1 = selectSVD(graphMatrix,n_components=nBlocks+1,
                                 n_elbows=self.n_elbows,algorithm='randomized',)
            tmp1 = nBlocks + 1
            eigenValGraph = d1[0:tmp1]
            if nCov > nBlocks:
                hmax = eigenValGraph[0]/(singValCov[nBlocks-1]**2 - singValCov[nBlocks]**2) 
            else:
                hmax = eigenValGraph[0]/singValCov[nCov-1]**2 
            hmin = (eigenValGraph[nBlocks-1] - eigenValGraph[nBlocks])/singValCov[0]**2
        else:
            u1,d1,v1 = selectSVD(graphMatrix,n_components=nBlocks+1,
                                 n_elbows=self.n_elbows,algorithm='randomized',)
            tmp1 = nBlocks + 1
            eigenValGraph = d1[0:tmp1]**2
            if nCov > nBlocks :
                hmax = eigenValGraph[0]/(singValCov[nBlocks-1]**2 - singValCov[nBlocks]**2) 
            else:
                hmax = eigenValGraph[0]/singValCov[nCov-1]**2 
            hmin = (eigenValGraph[nBlocks-1] - eigenValGraph[nBlocks])/singValCov[0]**2
        return [hmax,hmin]

    def getCascClusters(self,graphMat, covariates,hmin,hmax, 
                        nBlocks,nPoints, rowNorm, assortative,CCA):
        
        hTuningSeq = np.linspace(hmax,hmin,nPoints)
        wcssVec = []

        orthoX = []
        orthoL = []
        if not CCA:  
            for i in range(nPoints):
                cascResults = self.getCascResults(graphMat, covariates, 
                hTuningSeq[i],nBlocks, rowNorm, assortative,CCA)
                orthoL.append(cascResults['orthoL'])
                orthoX.append(cascResults['orthoX'])
                wcssVec.append(cascResults['wcss'])

            hOpt = hTuningSeq[wcssVec.index(min(wcssVec))]
        else:
            hOpt =0
        hOptResults = self.getCascResults(graphMat, covariates, hOpt, nBlocks,
                                          rowNorm,assortative,CCA)
        return {'cluster':hOptResults['cluster'],'h':hOpt,
                'wcss':hOptResults['wcss'],'cascSvd':hOptResults['cascSvd']}

    def getOrtho(self,graphMat, covariates, cascSvdEVec, cascSvdEVal,h, nBlocks):
        orthoL=\
        cascSvdEVec[:, (nBlocks-1)].transpose().dot(graphMat).dot(\
                    cascSvdEVec[:,(nBlocks-1)]).transpose()/cascSvdEVal[(nBlocks-1)]
        orthoX=\
        h*(cascSvdEVec[:, (nBlocks-1)].transpose().dot(covariates).dot(\
                       covariates.transpose()).dot(cascSvdEVec[:,(nBlocks-1)]\
                                           )/cascSvdEVal[(nBlocks-1)])
        return [orthoL/(orthoL + orthoX),orthoX/(orthoL + orthoX)]
    

    def getCascSvd(self,graphMat, covariates, hTuningParam, nBlocks,
                   assortative,CCA):
        if CCA: 
            New_laplacian=np.dot(graphMat,covariates)
            u2,d2,v2 = selectSVD(New_laplacian,
                                 n_components=min(New_laplacian.shape),
                                 n_elbows=self.n_elbows,algorithm='full')	

        else:
            if assortative:
                def matmult(x, y,t=False):
                    if t:
                        res=np.dot(graphMat.transpose(),y)+\
                        hTuningParam*np.dot( covariates, np.dot(covariates.transpose(), y))
                    else:
                        res=np.dot(graphMat,y) + \
                        hTuningParam*np.dot( covariates, np.dot(covariates.transpose(), y))
                    return res
                

            else:
                def matmult(x, y,t=False):
                    if t:
                        res=np.dot(graphMat,np.dot(graphMat.transpose(),y))+\
                        hTuningParam*np.dot( covariates, np.dot(covariates.transpose(), y))
                    else:
                        res=np.dot(graphMat,np.dot(graphMat,y)) + hTuningParam*np.dot(
                                covariates, np.dot(covariates.transpose(), y))
                    return res
            
                
            
            u2,d2,v2,it,pr = self.irlb(graphMat,nBlocks+1,matmult)	
        
        eVec = u2[:, 0:nBlocks]
        eVal = d2[0:(nBlocks+1)]
    
        return {'eVec':eVec,'eVal':eVal}

    def getCascResults(self,graphMat, covariates, hTuningParam,nBlocks, rowNorm,assortative,CCA):
        cascSvd = self.getCascSvd(graphMat, covariates, hTuningParam, nBlocks, assortative,CCA)
        
        if rowNorm:
            cascSvd_tmp = cascSvd['eVec']/np.sqrt(sum(cascSvd['eVec']**2))
        else:
            cascSvd_tmp = cascSvd['eVec']
        kmeansResults=KMeans(n_clusters=nBlocks).fit(cascSvd_tmp)
        
        if not CCA:
            ortho = self.getOrtho(graphMat, covariates,
                                  cascSvd['eVec'], cascSvd['eVal'],hTuningParam, nBlocks)

            return {'orthoL':ortho[0],'orthoX':ortho[1],
                    'wcss':kmeansResults.inertia_,'cluster':kmeansResults.labels_,'cascSvd':cascSvd_tmp}
    
        else:
            return {'orthoL':None,'orthoX':None,
                    'wcss':kmeansResults.inertia_,'cluster':kmeansResults.labels_,'cascSvd':cascSvd_tmp}
    
    
    
    def mult(self,A, x, t=False):
        assert x.ndim == 1
        if(sp.issparse(A)):
            if(t):
                return(sp.csr_matrix(x).dot(A).transpose().todense().A[:, 0])
            return(A.dot(sp.csr_matrix(x).transpose()).todense().A[:, 0])
        if(t):
            return np.asarray(A.transpose().dot(x)).ravel()
        return np.asarray(A.dot(x)).ravel()
    
    def orthog(self,Y, X):
        """Orthogonalize a vector or matrix Y against the columns of the matrix X.
        This function requires that the column dimension of Y is less than X and
        that Y and X have the same number of rows.
        """
        dotY = Y.dot(X)
        return (Y - self.mult(X, dotY))
    
    # Simple utility function used to check linear dependencies during computation:
    def invcheck(self,x):
        eps2 = 2 * np.finfo(np.float).eps
        if(x > eps2):
            return 1.0 / x
        else:
            warnings.warn(
                "Ill-conditioning encountered, result accuracy may be poor")
            return 0.0
    
    def irlb(self,A, n,matmult,tol=0.0001, maxit=50, center=None, scale=None, random_state=0):
        """Estimate a few of the largest singular values and corresponding singular
        vectors of matrix using the implicitly restarted Lanczos bidiagonalization
        method of Baglama and Reichel, see:
        Augmented Implicitly Restarted Lanczos Bidiagonalization Methods,
        J. Baglama and L. Reichel, SIAM J. Sci. Comput. 2005
        Keyword arguments:
        tol   -- An estimation tolerance. Smaller means more accurate estimates.
        maxit -- Maximum number of Lanczos iterations allowed.
        Given an input matrix A of dimension j * k, and an input desired number
        of singular values n, the function returns a tuple X with five entries:
        X[0] A j * nu matrix of estimated left singular vectors.
        X[1] A vector of length nu of estimated singular values.
        X[2] A k * nu matrix of estimated right singular vectors.
        X[3] The number of Lanczos iterations run.
        X[4] The number of matrix-vector products run.
        The algorithm estimates the truncated singular value decomposition:
        A.dot(X[2]) = X[0]*X[1].
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
        if(min(m, n) < 2):
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
    
        V = np.zeros((n, m_b)) # Approximate right vectors
        W = np.zeros((m, m_b)) # Approximate left vectors
        F = np.zeros((n, 1)) # Residual vector
        B = np.zeros((m_b, m_b)) #Bidiagonal approximation
    
        V[:, 0] = np.random.randn(n)  # Initial vector
        V[:, 0] = V[:, 0] / np.linalg.norm(V)
    
        while(it < maxit):
            if(it > 0):
                j = k
    
            VJ = V[:, j]
    
            # apply scaling
            if scale is not None:
                VJ = VJ / scale
    
            W[:, j] = matmult(A, VJ)
#            print(W,'VJ',VJ)
            mprod += 1
    
            # apply centering
            # R code: W[, j_w] <- W[, j_w] - ds * drop(cross(dv, VJ)) * du
            if center is not None:
                W[:, j] = W[:, j] - np.dot(center, VJ)
    
            if(it > 0):
                # NB W[:,0:j] selects columns 0,1,...,j-1
                W[:, j] = self.orthog(W[:, j], W[:, 0:j])
            s = np.linalg.norm(W[:, j])
            sinv = self.invcheck(s)
            W[:, j] = sinv * W[:, j]
    
            # Lanczos process
            while(j < m_b):
                F = matmult(A, W[:, j],t=True)
#                print(F)
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
                F = self.orthog(F, V[:, 0:(j + 1)])
                fn = np.linalg.norm(F)
                fninv = self.invcheck(fn)
                F = fninv * F
                if(j < m_b - 1):
                    V[:, j + 1] = F
                    B[j, j] = s
                    B[j, j + 1] = fn
                    VJp1 = V[:, j + 1]
    
                    # apply scaling
                    if scale is not None:
                        VJp1 = VJp1 / scale
    
                    W[:, j + 1] = matmult(A, VJp1)
#                    print(W)
                    mprod += 1
    
                    # apply centering
                    # R code: W[, jp1_w] <- W[, jp1_w] - ds * drop(cross(dv, VJP1)) * du
                    if center is not None:
                        W[:, j + 1] = W[:, j + 1] - np.dot(center, VJp1)
    
                    # One step of classical Gram-Schmidt...
                    W[:, j + 1] = W[:, j + 1] - fn * W[:, j]
                    # ...with full reorthogonalization
                    W[:, j + 1] =self. orthog(W[:, j + 1], W[:, 0:(j + 1)])
                    s = np.linalg.norm(W[:, j + 1])
                    sinv = self.invcheck(s)
                    W[:, j + 1] = sinv * W[:, j + 1]
                else:
                    B[j, j] = s
                j += 1
            # End of Lanczos process
            S = np.linalg.svd(B)
            R = fn * S[0][m_b - 1, :]  # Residuals
            if(it < 1):
                smax = S[1][0]  # Largest Ritz value
            else:
                smax = max((S[1][0], smax))
    
            conv = sum(np.abs(R[0:nu]) < tol * smax)
            if(conv < nu):  # Not coverged yet
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
        return((U, S[1][0:nu], V, it, mprod))