# sbm.py
# Created by Vikram Chandrashekhar on 2018-02-28.
# Email: vikramc@jhmi.edu

from ..embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from ..cluster import GaussianCluster
import numpy as np
from itertools import permutations


def get_block_probabilities(A, vertex_labels):
    """
    Get the probabilities associated with edges given 
    the block membership of each vertex.

    Parameters
    ----------
    A : array_like
            Input adjacency matrix for which parameters are estimated.
    vertex_labels : array_like
            block membership of each vertex
    """
    K = len(np.unique(vertex_labels))
    idx = [(np.nonzero(vertex_labels == i)[0], i) for i in np.unique(vertex_labels)]
    P_hat = np.zeros((K, K))
    for i, j in idx:
        P_hat[j, j] = np.mean(A[i, i.min() : i.max()])
    for i, j in permutations(idx, 2):
        P_hat[i[1], j[1]] = np.mean(A[i[0], j[0].min() : j[0].max()])
    return P_hat


def get_block_degrees(A, vertex_labels):
    """
    Get the degree for each vertex normalized by the total
    within-block degree.

    Parameters
    ----------
    A : array_like
            Input adjacency matrix for which parameters are estimated.
    vertex_labels : array_like
            block membership of each vertex
    """
    deg = np.sum(A, axis=1)
    total_degree0 = np.sum(deg[vertex_labels == 0])
    total_degree1 = np.sum(deg[vertex_labels == 1])
    deg[vertex_labels == 0] /= total_degree0
    deg[vertex_labels == 1] /= total_degree1
    return deg


def estimate_sbm_parameters(A, K, directed=False):
    """
    Estimate parameters for given graph under the Stochastic Block Model.
    
    The estimation is performed using Adjacency Spectral Embedding [1].

    Parameters
    ----------
    A : array_like
            Input adjacency matrix for which parameters are estimated.
    K : int
        Number of blocks in your graph.

    See Also
    --------
    graspy.embed.AdjacencySpectralEmbedding

    Returns
    -------
    n_hat : array-like, shape (n_vertices,)
            Estimated block membership for each vertex.
    P_hat : array-like, shape (K,K)
            Estimated probability of connection within and across blocks.

     References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012
    """
    if directed:
        # get both the left and right latent positions
        latent_positions = AdjacencySpectralEmbed().fit_transform(A)
        # but use only the left ones for clustering
        ase = latent_positions[0]
    else:
        ase = AdjacencySpectralEmbed().fit_transform(A)
    gclust = GaussianCluster(K).fit(ase)
    n_hat = gclust.predict(ase)
    P_hat = get_block_probabilities(A, n_hat)
    return n_hat, P_hat


def estimate_dcsbm_parameters(A, K, directed=False):
    """
    Estimate parameters for given graph under the Degree-Corrected Stochastic Block Model.
    
    The estimation is performed using Regularized Laplacian Spectral Embedding [1].

    Parameters
    ----------
    A : array_like
        Input adjacency matrix for which parameters are estimated.
    K : int
        Number of blocks in your graph.

    Returns
    -------
    n_hat : array-like, shape (n_vertices,)
            Estimated block membership for each vertex.
    P_hat : array-like, shape (K,K)
            Estimated probability of connection within and across blocks.
    theta_hat : array_like, shape (n_vertices,)
            Estimated vertex degree parameters assuming the sum of this
            parameter within each  block is 1.

    See Also
    --------
    graspy.embed.LaplacianSpectralEmbedding

    References
    ----------
    .. [1] Qin, Tai, and Karl Rohe. "Regularized spectral clustering
           under the degree-corrected stochastic blockmodel." In Advances
           in Neural Information Processing Systems, pp. 3120-3128. 2013.
    """
    if directed:
        raise("Laplacian not implemented/defined for directed graphs")
    lse = LaplacianSpectralEmbed(form="R-DAD", n_components=K).fit_transform(A)
    gclust = GaussianCluster(K).fit(lse)
    n_hat = gclust.predict(lse)
    P_hat = get_block_probabilities(A, n_hat)
    theta_hat = get_block_degrees(A, n_hat)
    return n_hat, P_hat, theta_hat
