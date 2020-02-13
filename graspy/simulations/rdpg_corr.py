import numpy as np
from graspy.simulations import p_from_latent, sample_edges_corr


def rdpg_corr(X, Y, r, rescale, directed, loops):
    """
    Samples a random graph based on the latent positions in X (and 
    optionally in Y)
    If only X :math:`\in\mathbb{R}^{n\times d}` is given, the P matrix is calculated as
    :math:`P = XX^T`. If X, Y :math:`\in\mathbb{R}^{n\times d}` is given, then 
    :math:`P = XY^T`. These operations correspond to the dot products between a set of 
    latent positions, so each row in X or Y represents the latent positions in  
    :math:`\mathbb{R}^{d}` for a single vertex in the random graph 
    Note that this function may also rescale or clip the resulting P 
    matrix to get probabilities between 0 and 1, or remove loops.
    A binary random graph is then sampled from the P matrix described 
    by X (and possibly Y).
    Read more in the :ref:`tutorials <simulations_tutorials>`

    Parameters
    ----------
    X: np.ndarray, shape (n_vertices, n_dimensions)
        latent position from which to generate a P matrix
        if Y is given, interpreted as the left latent position

    Y: np.ndarray, shape (n_vertices, n_dimensions) or None, optional
        right latent position from which to generate a P matrix

    r: float
        The value of the correlation between the same vertices in two graphs.

    rescale: boolean, optional (default=True)
        when rescale is True, will subtract the minimum value in 
        P (if it is below 0) and divide by the maximum (if it is
        above 1) to ensure that P has entries between 0 and 1. If
        False, elements of P outside of [0, 1] will be clipped

    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
        
    loops: boolean, optional (default=True)
        If False, no edges will be sampled in the diagonal. Diagonal elements in P 
        matrix are removed prior to rescaling (see above) which may affect behavior.
        Otherwise, edges are sampled in the diagonal.

    Returns
    -------
    G1: ndarray (n_vertices, n_vertices)
        A matrix representing the probabilities of connections between 
        vertices in a random graph based on their latent positions
        
    G2: ndarray (n_vertices, n_vertices)
        A matrix representing the probabilities of connections between 
        vertices in a random graph based on their latent positions

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012
    
    Examples
    --------
    >>> np.random.seed(1234)
    Generate random latent positions using 2-dimensional Dirichlet distribution.
    >>> X = np.random.dirichlet([1, 1], size=5)
    Sample a binary RDPG pair using sampled latent positions.
    >>> rdpg_corr(X,Y=None,0.3, rescale=False, directed=False, loops=False)
    array([[0., 1., 0., 1., 0.],
           [1., 0., 0., 1., 1.],
           [0., 0., 0., 0., 0.],
           [1., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0.]]), array([[0., 1., 0., 1., 0.],
           [1., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.]])
    """
    # check r
    if not np.issubdtype(type(r), np.floating):
        raise TypeError("r is not of type float.")
    elif r < -1 or r > 1:
        msg = "r must between -1 and 1."
        raise ValueError(msg)

    # check directed and loops
    if type(directed) is not bool:
        raise TypeError("directed is not of type bool.")
    if type(loops) is not bool:
        raise TypeError("loops is not of type bool.")

    # check dimensions of X and Y
    if Y != None:
        if type(X) is not np.ndarray or type(Y) is not np.ndarray:
            raise TypeError("Latent positions must be numpy.ndarray")
        if X.ndim != 2 or Y.ndim != 2:
            raise ValueError(
                "Latent positions must have dimension 2 (n_vertices, n_dimensions)"
            )
        if X.shape != Y.shape:
            raise ValueError("Dimensions of latent positions X and Y must be the same")
    if Y is None:
        Y = X

    P = p_from_latent(X, Y, rescale=rescale, loops=loops)
    n = np.size(P[0])
    R = r * np.ones((n, n))
    G1, G2 = sample_edges_corr(P, R, directed=directed, loops=loops)
    return G1, G2
