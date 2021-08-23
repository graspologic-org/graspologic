# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np

from graspologic.simulations import p_from_latent, sample_edges_corr


def rdpg_corr(X, Y, r, rescale=False, directed=False, loops=False):
    r"""
    Samples a random graph pair based on the latent positions in X (and
    optionally in Y)
    If only X :math:`\in\mathbb{R}^{n\times d}` is given, the P matrix is calculated as
    :math:`P = XX^T`. If X, Y :math:`\in\mathbb{R}^{n\times d}` is given, then
    :math:`P = XY^T`. These operations correspond to the dot products between a set of
    latent positions, so each row in X or Y represents the latent positions in
    :math:`\mathbb{R}^{d}` for a single vertex in the random graph.
    Note that this function may also rescale or clip the resulting P
    matrix to get probabilities between 0 and 1, or remove loops.
    A binary random graph is then sampled from the P matrix described
    by X (and possibly Y).

    Read more in the `Correlated Random Dot Product Graph (RDPG) Graph Pair Tutorial
    <https://microsoft.github.io/graspologic/tutorials/simulations/rdpg_corr.html>`_

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
        when ``rescale`` is True, will subtract the minimum value in
        P (if it is below 0) and divide by the maximum (if it is
        above 1) to ensure that P has entries between 0 and 1. If
        False, elements of P outside of [0, 1] will be clipped.

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
    .. [1] Vince Lyzinski, Donniell E Fishkind profile imageDonniell E. Fishkind, Carey E Priebe.
       "Seeded graph matching for correlated Erdös-Rényi graphs".
       The Journal of Machine Learning Research, January 2014

    Examples
    --------
    >>> np.random.seed(1234)
    >>> X = np.random.dirichlet([1, 1], size=5)
    >>> Y = None

    Generate random latent positions using 2-dimensional Dirichlet distribution.
    Then sample a correlated RDPG graph pair:

    >>> rdpg_corr(X, Y, 0.3, rescale=False, directed=False, loops=False)
    (array([[0., 1., 0., 1., 0.],
           [1., 0., 0., 1., 1.],
           [0., 0., 0., 0., 0.],
           [1., 1., 0., 0., 0.],
           [0., 1., 0., 0., 0.]]), array([[0., 1., 0., 1., 0.],
           [1., 0., 0., 0., 1.],
           [0., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 1., 0., 0., 0.]]))
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
    n = P.shape[0]
    R = np.full((n, n), r)
    G1, G2 = sample_edges_corr(P, R, directed=directed, loops=loops)
    return G1, G2
