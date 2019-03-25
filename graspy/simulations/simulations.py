import numpy as np
import warnings

from ..utils import symmetrize


def cartprod(*arrays):
    N = len(arrays)
    return np.transpose(
        np.meshgrid(*arrays, indexing="ij"), np.roll(np.arange(N + 1), -1)
    ).reshape(-1, N)


def sample_edges(P, directed=False, loops=False):
    """
    Gemerates a binary random graph based on the P matrix provided

    Each element in P represents the probability of a connection between 
    a vertex indexed by the row i and the column j. 

    Parameters
    ----------
    P: np.ndarray, shape (n_vertices, n_vertices)
        Matrix of probabilities (between 0 and 1) for a random graph
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.

    Returns
    -------
    A: ndarray (n_vertices, n_vertices)
        Binary adjacency matrix the same size as P representing a random
        graph

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012
    """
    if type(P) is not np.ndarray:
        raise TypeError("P must be numpy.ndarray")
    if len(P.shape) != 2:
        raise ValueError("P must have dimension 2 (n_vertices, n_dimensions)")
    if P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix")
    if not directed:
        # can cut down on sampling by ~half
        triu_inds = np.triu_indices(P.shape[0])
        samples = np.random.binomial(1, P[triu_inds])
        A = np.zeros_like(P)
        A[triu_inds] = samples
        A = symmetrize(A)
    else:
        A = np.random.binomial(1, P)

    if loops:
        return A
    else:
        return A - np.diag(np.diag(A))


def er_np(n, p, directed=False, loops=False, wt=1, wtargs=None):
    r"""
    Samples a Erdos Renyi (n, p) graph with specified edge probability.

    Erdos Renyi (n, p) graph is a simple graph with n vertices and a probability
    p of edges being connected.

    Parameters
    ----------
    n: int
        Number of vertices
    p: float
        Probability of an edge existing between two vertices, between 0 and 1.
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.
    wt: object, optional (default=1)
        Weight function for each of the edges, taking only a size argument. 
        This weight function will be randomly assigned for selected edges. 
        If 1, graph produced is binary.
    wtargs: dictionary, optional (default=None)
        Optional arguments for parameters that can be passed
        to weight function ``wt``.

    Returns
    -------
    A : ndarray, shape (n, n)
        Sampled adjacency matrix
    """
    if not np.issubdtype(type(p), np.floating):
        raise TypeError("p is not of type float.")
    elif p < 0:
        msg = "You have passed a probability, {}, less than 0."
        msg = msg.format(float(p))
        raise ValueError(msg)
    elif p > 1:
        msg = "You have passed a probability, {}, greater than 1."
        msg = msg.format(float(p))
        raise ValueError(msg)
    if not np.issubdtype(type(n), np.integer):
        raise TypeError("n is not of type int.")
    elif n <= 0:
        msg = "n must be > 0."
        raise ValueError(msg)
    if type(directed) is not bool:
        raise TypeError("directed is not of type bool.")
    if type(loops) is not bool:
        raise TypeError("loops is not of type bool.")

    # check weight function
    if not np.issubdtype(type(wt), np.number):
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")

    probs = np.ones((n, n)) * p
    A = sample_edges(probs, directed, loops)

    if not np.issubdtype(type(wt), np.number):
        weights = wt(size=int(A.sum()), **wtargs)
        A[A == 1] = weights
    else:
        A *= wt

    if not directed:
        A = symmetrize(A)

    return A


def er_nm(n, m, directed=False, loops=False, wt=1, wtargs=None):
    r"""
    Samples an Erdos Renyi (n, m) graph with specified number of edges.

    Erdos Renyi (n, m) graph is a simple graph with n vertices and exactly m
    number of total edges.

    Parameters
    ----------
    n: int
        Number of vertices
    m: int
        Number of edges, a value between 1 and :math:`n^2`.
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.
    wt: object, optional (default=1)
        Weight function for each of the edges, taking only a size argument. 
        This weight function will be randomly assigned for selected edges. 
        If 1, graph produced is binary.
    wtargs: dictionary, optional (default=None)
        Optional arguments for parameters that can be passed
        to weight function ``wt``.

    Returns
    -------
    A: ndarray, shape (n, n)
        Sampled adjacency matrix

    Examples
    --------
    >>> n = 100
    >>> m = 20
    >>> wt = np.random.uniform
    >>> wtargs = dict(low=1, high=2)
    >>> A = weighted_er_nm(n, m, wt=wt, wtargs=wtargs)
    """
    if not np.issubdtype(type(m), np.integer):
        raise TypeError("m is not of type int.")
    elif m <= 0:
        msg = "m must be > 0."
        raise ValueError(msg)
    if not np.issubdtype(type(n), np.integer):
        raise TypeError("n is not of type int.")
    elif n <= 0:
        msg = "n must be > 0."
        raise ValueError(msg)
    if type(directed) is not bool:
        raise TypeError("directed is not of type bool.")
    if type(loops) is not bool:
        raise TypeError("loops is not of type bool.")

    # check weight function
    if not np.issubdtype(type(wt), np.integer):
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")

    # compute max number of edges to sample
    if loops:
        if directed:
            max_edges = n ** 2
            msg = "n^2"
        else:
            max_edges = n * (n + 1) // 2
            msg = "n(n+1)/2"
    else:
        if directed:
            max_edges = n * (n - 1)
            msg = "n(n-1)"
        else:
            max_edges = n * (n - 1) // 2
            msg = "n(n-1)/2"
    if m > max_edges:
        msg = "You have passed a number of edges, {}, exceeding {}, {}."
        msg = msg.format(m, msg, max_edges)
        raise ValueError(msg)

    A = np.zeros((n, n))
    # check if directedness is desired
    if directed:
        if loops:
            # use all of the indices
            idx = np.where(np.logical_not(A))
        else:
            # use only the off-diagonal indices
            idx = np.where(~np.eye(n, dtype=bool))
    else:
        # use upper-triangle indices, and ignore diagonal according
        # to loops argument
        idx = np.triu_indices(n, k=int(loops == False))

    # get idx in 1d coordinates by ravelling
    triu = np.ravel_multi_index(idx, A.shape)
    # choose M of them
    triu = np.random.choice(triu, size=m, replace=False)
    # unravel back
    triu = np.unravel_index(triu, A.shape)
    # check weight function
    if not np.issubdtype(type(wt), np.number):
        wt = wt(size=m, **wtargs)
    A[triu] = wt

    if not directed:
        A = symmetrize(A)

    return A


def sbm(n, p, directed=False, loops=False, wt=1, wtargs=None, dc=None, dc_kws={}):
    """
    Samples a graph from the stochastic block model (SBM). 

    SBM produces a graph with specified communities, in which each community can
    have different sizes and edge probabilities. 

    Parameters
    ----------
    n: list of int, shape (n_communities)
        Number of vertices in each community. Communities are assigned n[0], n[1], ...
    p: array-like, shape (n_communities, n_communities)
        Probability of an edge between each of the communities, where p[i, j] indicates 
        the probability of a connection between edges in communities [i, j]. 
        0 < p[i, j] < 1 for all i, j.
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=False)
        If False, no edges will be sampled in the diagonal. Otherwise, edges
        are sampled in the diagonal.
    wt: object or array-like, shape (n_communities, n_communities)
        if Wt is an object, a weight function to use globally over
        the sbm for assigning weights. 1 indicates to produce a binary
        graph. If Wt is an array-like, a weight function for each of
        the edge communities. Wt[i, j] corresponds to the weight function
        between communities i and j. If the entry is a function, should
        accept an argument for size. An entry of Wt[i, j] = 1 will produce a
        binary subgraph over the i, j community.
    wtargs: dictionary or array-like, shape (n_communities, n_communities)
        if Wt is an object, Wtargs corresponds to the trailing arguments
        to pass to the weight function. If Wt is an array-like, Wtargs[i, j] 
        corresponds to trailing arguments to pass to Wt[i, j].
    dc: function or array-like, shape (n_vertices) or (n_communities), optional
        `dc` is used to generate a degree-corrected stochastic block model [1] in 
        which each node in the graph has a parameter to specify its expected degree
        relative to other nodes within its community.

        - function: 
            should generate a non-negative number to be used as a degree correction to 
            create a heterogenous degree distribution. A weight will be generated for
            each vertex, normalized so that the sum of weights in each block is 1. 
        - array-like of functions, shape (n_communities): 
            Each function will generate the degree distribution for its respective 
            community. 
        - array-like of scalars, shape (n_vertices): 
            The weights in each block should sum to 1; otherwise, they will be normalized
            and a warning will be thrown. The scalar associated with each vertex is the 
            node's relative expected degree within its community. 
    
    dc_kws: dictionary or array-like, shape (n_communities), optional
        Ignored if `dc` is none or array of scalar.
        If `dc` is a function, `dc_kws` corresponds to its named arguments. 
        If `dc` is an array-like of functions, `dc_kws` should be an array-like, shape 
        (n_communities), of dictionary. Each dictionary is the named arguments 
        for the corresponding function for that community. 
        If not specified, in either case all functions will assume their default 
        parameters.

    References
    ----------
    .. [1] Tai Qin and Karl Rohe. "Regularized spectral clustering under the 
        Degree-Corrected Stochastic Blockmodel," Advances in Neural Information 
        Processing Systems 26, 2013

    Returns
    -------
    A: ndarray, shape (sum(n), sum(n))
        Sampled adjacency matrix
        
    """
    # Check n
    if not isinstance(n, (list, np.ndarray)):
        msg = "n must be a list or np.array, not {}.".format(type(n))
        raise TypeError(msg)
    else:
        n = np.array(n)
        if not np.issubdtype(n.dtype, np.integer):
            msg = "There are non-integer elements in n"
            raise ValueError(msg)

    # Check p
    if not isinstance(p, (list, np.ndarray)):
        msg = "p must be a list or np.array, not {}.".format(type(p))
        raise TypeError(msg)
    else:
        p = np.array(p)
        if not np.issubdtype(p.dtype, np.number):
            msg = "There are non-numeric elements in p"
            raise ValueError(msg)
        elif p.shape != (n.size, n.size):
            msg = "p is must have shape len(n) x len(n), not {}".format(p.shape)
            raise ValueError(msg)
        elif np.any(p < 0) or np.any(p > 1):
            msg = "Values in p must be in between 0 and 1."
            raise ValueError(msg)

    # Check wt and wtargs
    if not np.issubdtype(type(wt), np.number) and not callable(wt):
        if not isinstance(wt, (list, np.ndarray)):
            msg = "wt must be a numeric, list, or np.array, not {}".format(type(wt))
            raise TypeError(msg)
        if not isinstance(wtargs, (list, np.ndarray)):
            msg = "wtargs must be a numeric, list, or np.array, not {}".format(
                type(wtargs)
            )
            raise TypeError(msg)

        wt = np.array(wt, dtype=object)
        wtargs = np.array(wtargs, dtype=object)
        # if not number, check dimensions
        if wt.shape != (n.size, n.size):
            msg = "wt must have size len(n) x len(n), not {}".format(wt.shape)
            raise ValueError(msg)
        if wtargs.shape != (n.size, n.size):
            msg = "wtargs must have size len(n) x len(n), not {}".format(wtargs.shape)
            raise ValueError(msg)
        # check if each element is a function
        for element in wt.ravel():
            if not callable(element):
                msg = "{} is not a callable function.".format(element)
                raise TypeError(msg)
    else:
        wt = np.full(p.shape, wt, dtype=object)
        wtargs = np.full(p.shape, wtargs, dtype=object)

    # Check directed
    if not directed:
        if np.any(p != p.T):
            raise ValueError("Specified undirected, but P is directed.")
        if np.any(wt != wt.T):
            raise ValueError("Specified undirected, but Wt is directed.")
        if np.any(wtargs != wtargs.T):
            raise ValueError("Specified undirected, but Wtargs is directed.")

    K = len(n)  # the number of communities
    counter = 0
    # get a list of community indices
    cmties = []
    for i in range(0, K):
        cmties.append(range(counter, counter + n[i]))
        counter += n[i]

    # Check degree-corrected input parameters
    if callable(dc):
        # Check that the paramters are a dict
        if not isinstance(dc_kws, dict):
            msg = "dc_kws must be of type dict not{}".format(type(dc_kws))
            raise TypeError(msg)
        # Create the probability matrix for each vertex
        dcProbs = np.array([dc(**dc_kws) for _ in range(0, sum(n))], dtype="float")
        for indices in cmties:
            dcProbs[indices] /= sum(dcProbs[indices])
    elif isinstance(dc, (list, np.ndarray)) and np.issubdtype(
        np.array(dc).dtype, np.number
    ):
        dcProbs = np.array(dc)
        # Check size and element types
        if not np.issubdtype(dcProbs.dtype, np.number):
            msg = "There are non-numeric elements in dc, {}".format(dcProbs.dtype)
            raise ValueError(msg)
        elif dcProbs.shape != (sum(n),):
            msg = "dc must have size equal to the number of vertices {0}, not {1}".format(
                sum(n), dcProbs.shape
            )
            raise ValueError(msg)
        elif np.any(dcProbs < 0):
            msg = "Values in dc cannot be negative."
            raise ValueError(msg)
        # Check that probabilities sum to 1 in each block
        for i in range(0, K):
            if not np.isclose(sum(dcProbs[cmties[i]]), 1, atol=1.0e-8):
                msg = "Block {} probabilities should sum to 1, normalizing...".format(i)
                warnings.warn(msg, UserWarning)
                dcProbs[cmties[i]] /= sum(dcProbs[cmties[i]])
    elif isinstance(dc, (list, np.ndarray)) and all(callable(f) for f in dc):
        dcFuncs = np.array(dc)
        if dcFuncs.shape != (len(n),):
            msg = "dc must have size equal to the number of blocks {0}, not {1}".format(
                len(n), dcFuncs.shape
            )
            raise ValueError(msg)
        # Check that the parameters type, length, and type
        if not isinstance(dc_kws, (list, np.ndarray)):
            # Allows for nonspecification of default parameters for all functions
            if dc_kws == {}:
                dc_kws = [{} for _ in range(0, len(n))]
            else:
                msg = "dc_kws must be of type list or np.ndarray, not {}".format(
                    type(dc_kws)
                )
                raise TypeError(msg)
        elif not len(dc_kws) == len(n):
            msg = "dc_kws must have size equal to the number of blocks {0}, not {1}".format(
                len(n), len(dc_kws)
            )
            raise ValueError(msg)
        elif not all(type(kw) == dict for kw in dc_kws):
            msg = "dc_kws elements must all be of type dict"
            raise TypeError(msg)
        # Create the probability matrix for each vertex
        dcProbs = np.array(
            [
                dcFunc(**kws)
                for dcFunc, kws, size in zip(dcFuncs, dc_kws, n)
                for _ in range(0, size)
            ],
            dtype="float",
        )
        for indices in cmties:
            dcProbs[indices] /= sum(dcProbs[indices])
    elif dc is not None:
        msg = "dc must be a function or a list or np.array of numbers or callable functions, not {}".format(
            type(dc)
        )
        raise ValueError(msg)

    # End Checks, begin simulation
    A = np.zeros((sum(n), sum(n)))

    for i in range(0, K):
        if directed:
            jrange = range(0, K)
        else:
            jrange = range(i, K)
        for j in jrange:
            block_wt = wt[i, j]
            block_wtargs = wtargs[i, j]
            block_p = p[i, j]
            # identify submatrix for community i, j
            # cartesian product to identify edges for community i,j pair
            cprod = cartprod(cmties[i], cmties[j])
            # get idx in 1d coordinates by ravelling
            triu = np.ravel_multi_index((cprod[:, 0], cprod[:, 1]), A.shape)
            pchoice = np.random.uniform(size=len(triu))
            if dc is not None:
                # (v1,v2) connected with probability p*k_i*k_j*dcP[v1]*dcP[v2]
                num_edges = sum(pchoice < block_p)
                edge_dist = dcProbs[cprod[:, 0]] * dcProbs[cprod[:, 1]]
                # If the number edges greater than support of dc distribution, pick fewer edges
                if num_edges > sum(edge_dist > 0):
                    msg = "More edges sampled than nonzero pairwise dc entries. Picking fewer edges"
                    warnings.warn(msg, UserWarning)
                    num_edges = sum(edge_dist > 0)
                triu = np.random.choice(
                    triu, size=num_edges, replace=False, p=edge_dist
                )
            else:
                # connected with probability p
                triu = triu[pchoice < block_p]
            if type(block_wt) is not int:
                block_wt = block_wt(size=len(triu), **block_wtargs)
            triu = np.unravel_index(triu, A.shape)
            A[triu] = block_wt

    if not loops:
        A = A - np.diag(np.diag(A))
    if not directed:
        A = symmetrize(A)
    return A


def rdpg(X, Y=None, rescale=True, directed=False, loops=True, wt=1, wtargs=None):
    r"""
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

    Parameters
    ----------
    X: np.ndarray, shape (n_vertices, n_dimensions)
        latent position from which to generate a P matrix
        if Y is given, interpreted as the left latent position
    Y: np.ndarray, shape (n_vertices, n_dimensions) or None, optional
        right latent position from which to generate a P matrix
    rescale: boolean, optional (default=True)
        when rescale is True, will subtract the minimum value in 
        P (if it is below 0) and divide by the maximum (if it is
        above 1) to ensure that P has entries between 0 and 1. If
        False, elements of P outside of [0, 1] will be clipped
    directed: boolean, optional (default=False)
        If False, output adjacency matrix will be symmetric. Otherwise, output adjacency
        matrix will be asymmetric.
    loops: boolean, optional (default=True)
        If False, no edges will be sampled in the diagonal. Diagonal elements in P matrix
        are removed prior to rescaling (see above) which may affect behavior. Otherwise,
        edges are sampled in the diagonal.
    wt: object, optional (default=1)
        Weight function for each of the edges, taking only a size argument. 
        This weight function will be randomly assigned for selected edges. 
        If 1, graph produced is binary.
    wtargs: dictionary, optional (default=None)
        Optional arguments for parameters that can be passed
        to weight function ``wt``.

    Returns
    -------
    A: ndarray (n_vertices, n_vertices)
        A matrix representing the probabilities of connections between 
        vertices in a random graph based on their latent positions

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012
    
    """
    P = p_from_latent(X, Y, rescale=rescale, loops=loops)
    A = sample_edges(P, directed=directed, loops=loops)

    # check weight function
    if (not np.issubdtype(type(wt), np.integer)) and (
        not np.issubdtype(type(wt), np.floating)
    ):
        if not callable(wt):
            raise TypeError("You have not passed a function for wt.")

    if not np.issubdtype(type(wt), np.number):
        wts = wt(size=(np.count_nonzero(A)), **wtargs)
        A[A > 0] = wts
    else:
        A *= wt
    return A


def p_from_latent(X, Y=None, rescale=True, loops=True):
    r"""
    Gemerates a matrix of connection probabilities for a random graph
    based on a set of latent positions

    If only X is given, the P matrix is calculated as :math:`P = XX^T`
    If X and Y is given, then :math:`P = XY^T`
    These operations correspond to the dot products between a set of latent
    positions, so each row in X or Y represents the latent positions in  
    :math:`\mathbb{R}^{num-columns}` for a single vertex in the random graph 
    Note that this function may also rescale or clip the resulting P 
    matrix to get probabilities between 0 and 1, or remove loops

    Parameters
    ----------
    X: np.ndarray, shape (n_vertices, n_dimensions)
        latent position from which to generate a P matrix
        if Y is given, interpreted as the left latent position
    Y: np.ndarray, shape (n_vertices, n_dimensions) or None, optional
        right latent position from which to generate a P matrix
    rescale: boolean, optional (default=True)
        when rescale is True, will subtract the minimum value in 
        P (if it is below 0) and divide by the maximum (if it is
        above 1) to ensure that P has entries between 0 and 1. If
        False, elements of P outside of [0, 1] will be clipped
    loops: boolean, optional (default=True)
        whether to allow elements on the diagonal (corresponding
        to self connections in a graph) in the returned P matrix. 
        If loops is False, these elements are removed prior to 
        rescaling (see above) which may affect behavior

    Returns
    -------
    P: ndarray (n_vertices, n_vertices)
        A matrix representing the probabilities of connections between 
        vertices in a random graph based on their latent positions

    References
    ----------
    .. [1] Sussman, D.L., Tang, M., Fishkind, D.E., Priebe, C.E.  "A
       Consistent Adjacency Spectral Embedding for Stochastic Blockmodel Graphs,"
       Journal of the American Statistical Association, Vol. 107(499), 2012
    
    """
    if Y is None:
        Y = X
    if type(X) is not np.ndarray or type(Y) is not np.ndarray:
        raise TypeError("Latent positions must be numpy.ndarray")
    if X.ndim != 2 or Y.ndim != 2:
        raise ValueError(
            "Latent positions must have dimension 2 (n_vertices, n_dimensions)"
        )
    if X.shape != Y.shape:
        raise ValueError("Dimensions of latent positions X and Y must be the same")
    P = X @ Y.T
    # should this be before or after the rescaling, could give diff answers
    if not loops:
        P = P - np.diag(np.diag(P))
    if rescale:
        if P.min() < 0:
            P = P - P.min()
        if P.max() > 1:
            P = P / P.max()
    else:
        P[P < 0] = 0
        P[P > 1] = 1
    return P
