# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from collections import namedtuple

import numpy as np
from hyppo.ksample import KSample
from joblib import Parallel, delayed
from scipy import stats
from sklearn.metrics.pairwise import PAIRED_DISTANCES, PAIRWISE_KERNEL_FUNCTIONS
from sklearn.utils import check_array, check_random_state

from ..align import SeedlessProcrustes, SignFlips
from ..embed import AdjacencySpectralEmbed, select_dimension
from ..utils import fit_plug_in_variance_estimator, import_graph

_VALID_DISTANCES = list(PAIRED_DISTANCES.keys())
_VALID_KERNELS = list(PAIRWISE_KERNEL_FUNCTIONS.keys())
_VALID_KERNELS.append("gaussian")  # can use hyppo's medial gaussian kernel too
_VALID_METRICS = _VALID_DISTANCES + _VALID_KERNELS

_VALID_TESTS = ["cca", "dcorr", "hhg", "rv", "hsic", "mgc"]

ldt_result = namedtuple("ldt_result", ("p_value", "sample_T_statistic", "misc_stats"))


def latent_distribution_test(
    A1,
    A2,
    test="dcorr",
    metric="euclidean",
    n_components=None,
    n_bootstraps=500,
    random_state=None,
    workers=None,
    size_correction=True,
    pooled=False,
    align_type="sign_flips",
    align_kws={},
    input_graph=True,
):
    """Two-sample hypothesis test for the problem of determining whether two random
    dot product graphs have the same distributions of latent positions.

    This test can operate on two graphs where there is no known matching
    between the vertices of the two graphs, or even when the number of vertices
    is different. Currently, testing is only supported for undirected graphs.

    Read more in the `Latent Distribution Two-Graph Testing Tutorial
    <https://microsoft.github.io/graspologic/tutorials/inference/latent_distribution_test.html>`_

    Parameters
    ----------
    A1, A2 : variable (see description of 'input_graph')
        The two graphs, or their embeddings to run a hypothesis test on.
        Expected variable type and shape depends on input_graph attribute

    test : str (default="dcorr")
        Backend hypothesis test to use, one of ["cca", "dcorr", "hhg", "rv", "hsic", "mgc"].
        These tests are typically used for independence testing, but here they
        are used for a two-sample hypothesis test on the latent positions of
        two graphs. See :class:`hyppo.ksample.KSample` for more information.

    metric : str or function (default="euclidean")
        Distance or a kernel metric to use, either a callable or a valid string.
        Kernel metrics (e.g. "gaussian") must be used with kernel-based HSIC test
        and distances (e.g. "euclidean") with all other tests. If a callable,
        then it should behave similarly to either
        :func:`sklearn.metrics.pairwise_distances` or to
        :func:`sklearn.metrics.pairwise.pairwise_kernels`.

        Valid strings for distance ``metric`` are, as defined in
        :func:`sklearn.metrics.pairwise_distances`,

            - From scikit-learn: [``"euclidean"``, ``"cityblock"``, ``"cosine"``,
              ``"l1"``, ``"l2"``, ``"manhattan"``].
            - From scipy.spatial.distance: [``"braycurtis"``, ``"canberra"``,
              ``"chebyshev"``, ``"correlation"``, ``"dice"``, ``"hamming"``,
              ``"jaccard"``, ``"kulsinski"``, ``"mahalanobis"``, ``"minkowski"``,
              ``"rogerstanimoto"``, ``"russellrao"``, ``"seuclidean"``,
              ``"sokalmichener"``, ``"sokalsneath"``, ``"sqeuclidean"``,
              ``"yule"``] See the documentation for :mod:`scipy.spatial.distance` for
              details on these metrics.

        Valid strings for kernel ``metric`` are, as defined in
        :func:`sklearn.metrics.pairwise.pairwise_kernels`,

            [``"additive_chi2"``, ``"chi2"``, ``"linear"``, ``"poly"``,
            ``"polynomial"``, ``"rbf"``,
            ``"laplacian"``, ``"sigmoid"``, ``"cosine"``]

        Note ``"rbf"`` and ``"gaussian"`` are the same metric, which will use
        an adaptively selected bandwidth.

    n_components : int or None (default=None)
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.
        See :func:`~graspologic.embed.select_svd` for more information.
        This argument is ignored if ``input_graph`` is False.

    n_bootstraps : int (default=200)
        Number of bootstrap iterations for the backend hypothesis test.
        See :class:`hyppo.ksample.KSample` for more information.

    random_state : {None, int, `~np.random.RandomState`, `~np.random.Generator`}
        This parameter defines the object to use for drawing random
        variates.
        If `random_state` is ``None`` the `~np.random.RandomState` singleton is
        used.
        If `random_state` is an int, a new ``RandomState`` instance is used,
        seeded with `random_state`.
        If `random_state` is already a ``RandomState`` or ``Generator``
        instance, then that object is used.
        Default is None.

    workers : int or None (default=None)
        Number of workers to use. If more than 1, parallelizes the code.
        Supply -1 to use all cores available. None is a marker for
        'unset' that will be interpreted as ``workers=1`` (sequential execution) unless
        the call is performed under a Joblib parallel_backend context manager that sets
        another value for ``workers``. See :class:joblib.Parallel for more details.

    size_correction : bool (default=True)
        Ignored when the two graphs have the same number of vertices. The test
        degrades in validity as the number of vertices of the two graphs
        diverge from each other, unless a correction is performed.

        - True
            Whenever the two graphs have different numbers of vertices,
            estimates the plug-in estimator for the variance and uses it to
            correct the embedding of the larger graph.
        - False
            Does not perform any modifications (not recommended).

    pooled : bool (default=False)
        Ignored whenever the two graphs have the same number of vertices or
        ``size_correction`` is set to False. In order to correct the adjacency
        spectral embedding used in the test, it is needed to estimate the
        variance for each of the latent position estimates in the larger graph,
        which requires to compute different sample moments. These moments can
        be computed either over the larger graph (False), or over both graphs
        (True). Setting it to True should not affect the behavior of the test
        under the null hypothesis, but it is not clear whether it has more
        power or less power under which alternatives. Generally not recomended,
        as it is untested and included for experimental purposes.

    align_type : str, {'sign_flips' (default), 'seedless_procrustes'} or None
        Random dot product graphs have an inherent non-identifiability,
        associated with their latent positions. Thus, two embeddings of
        different graphs may not be orthogonally aligned. Without this accounted
        for, two embeddings of different graphs may appear different, even
        if the distributions of the true latent positions are the same.
        There are several options in terms of how this can be addresssed:

        - 'sign_flips'
            A simple heuristic that flips the signs of one of the embeddings,
            if the medians of the two embeddings in that dimension differ from
            each other. See :class:`graspologic.align.SignFlips` for more
            information on this procedure. In the limit, this is guaranteed to
            lead to a valid test, as long as matrix :math:`X^T X`, where
            :math:`X` is the latent positions does not have repeated non-zero
            eigenvalues. This may, however, result in an invalid test in the
            finite sample case if the some eigenvalues are same or close.
        - 'seedless_procrustes'
            An algorithm that learns an orthogonal alignment matrix. This
            procedure is slower than sign flips, but is guaranteed to yield a
            valid test in the limit, and also makes the test more valid in some
            finite sample cases, in which the eigenvalues are very close to
            each other. See :class:`graspologic.align.SignFlips` for more information
            on the procedure.
        - None
            Do not use any alignment technique. This is strongly not
            recommended, as it may often result in a test that is not valid.

    align_kws : dict
        Keyword arguments for the aligner of choice, either
        :class:`graspologic.align.SignFlips` or
        :class:`graspologic.align.SeedlessProcrustes`, depending on the ``align_type``.
        See respective classes for more information.

    input_graph : bool (default=True)
        Flag whether to expect two full graphs, or the embeddings.

        - True
            This function expects graphs, either as NetworkX graph objects
            or as adjacency matrices, provided as ndarrays of size (n, n) and
            (m, m). They will be embedded using adjacency spectral embeddings.
        - False
            This function expects adjacency spectral embeddings of the graphs,
            they must be ndarrays of size (n, d) and (m, d), where
            d must be same. n_components attribute is ignored in this case.

    Returns
    ----------
    p_value : float
        The overall p value from the test.

    sample_T_statistic : float
        The observed difference between the embedded latent positions of the
        two input graphs.

    misc_stats : dictionary
        A collection of other statistics obtained from the latent position test

        - null_distribution : ndarray, shape (n_bootstraps,)
            The distribution of T statistics generated under the null.

        - n_components : int
            Number of embedding dimensions.

        - Q : array, size (d, d)
            Final orthogonal matrix, used to modify ``X``.

    References
    ----------
    .. [1] Tang, M., Athreya, A., Sussman, D. L., Lyzinski, V., & Priebe, C. E. (2017).
        "A nonparametric two-sample hypothesis testing problem for random graphs."
        Bernoulli, 23(3), 1599-1630.

    .. [2] Panda, S., Palaniappan, S., Xiong, J., Bridgeford, E., Mehta, R., Shen, C., & Vogelstein, J. (2019).
        "hyppo: A Comprehensive Multivariate Hypothesis Testing Python Package."
        arXiv:1907.02088.

    .. [3] Alyakin, A. A., Agterberg, J., Helm, H. S., Priebe, C. E. (2020).
       "Correcting a Nonparametric Two-sample Graph Hypothesis Test for Graphs with Different Numbers of Vertices"
       arXiv:2008.09434

    """

    # check test argument
    if not isinstance(test, str):
        msg = "test must be a str, not {}".format(type(test))
        raise TypeError(msg)
    elif test not in _VALID_TESTS:
        msg = "Unknown test {}. Valid tests are {}".format(test, _VALID_TESTS)
        raise ValueError(msg)

    # check metric argument
    if not isinstance(metric, str) and not callable(metric):
        msg = "Metric must be str or callable, not {}".format(type(metric))
        raise TypeError(msg)
    elif metric not in _VALID_METRICS and not callable(metric):
        msg = "Unknown metric {}. Valid metrics are {}, or a callable".format(
            metric, _VALID_METRICS
        )
        raise ValueError(msg)

    if metric in _VALID_DISTANCES:
        if test == "hsic":
            msg = (
                f"{test} is a kernel-based test, but {metric} "
                "is a distance. Use a different test or one of "
                f"the kernels: {_VALID_KERNELS} as a metric."
            )
            raise ValueError(msg)
    elif metric in _VALID_KERNELS:
        if test != "hsic":
            msg = (
                f"{test} is a distance-based test, but {metric} "
                "is a kernel. Use either a HSIC as the test or one of "
                f"the distances: {_VALID_DISTANCES} as a metric."
            )
            raise ValueError(msg)

    # check n_components argument
    if n_components is not None:
        if not isinstance(n_components, int):
            msg = "n_components must be an int, not {}.".format(type(n_components))
            raise TypeError(msg)

    # check n_bootstraps argument
    if not isinstance(n_bootstraps, int):
        msg = "n_bootstraps must be an int, not {}".format(type(n_bootstraps))
        raise TypeError(msg)
    elif n_bootstraps < 0:
        msg = "{} is invalid number of bootstraps, must be non-negative"
        raise ValueError(msg.format(n_bootstraps))

    # check workers argument
    if workers is not None and not isinstance(workers, (int, np.integer)):
        msg = "workers must be an int or None, not {}".format(type(workers))
        raise TypeError(msg)

    # check size_correction argument
    if not isinstance(size_correction, bool):
        msg = "size_correction must be a bool, not {}".format(type(size_correction))
        raise TypeError(msg)

    # check pooled argument
    if not isinstance(pooled, bool):
        msg = "pooled must be a bool, not {}".format(type(pooled))
        raise TypeError(msg)

    # check align_type argument
    if (not isinstance(align_type, str)) and (align_type is not None):
        msg = "align_type must be a string or None, not {}".format(type(align_type))
        raise TypeError(msg)
    align_types_supported = ["sign_flips", "seedless_procrustes", None]
    if align_type not in align_types_supported:
        msg = "supported align types are {}".format(align_types_supported)
        raise ValueError(msg)

    # check align_kws argument
    if not isinstance(align_kws, dict):
        msg = "align_kws must be a dictionary of keyword arguments, not {}".format(
            type(align_kws)
        )
        raise TypeError(msg)

    # check input_graph argument
    if not isinstance(input_graph, bool):
        msg = "input_graph must be a bool, not {}".format(type(input_graph))
        raise TypeError(msg)

    if input_graph:
        A1 = import_graph(A1)
        A2 = import_graph(A2)

        X1_hat, X2_hat = _embed(A1, A2, n_components)
    else:
        # check for nx objects, since they are castable to arrays,
        # but we don't want that
        if not isinstance(A1, np.ndarray):
            msg = (
                f"Embedding of the first graph is of type {type(A1)}, not "
                "np.ndarray. If input_graph is False, the inputs need to be "
                "adjacency spectral embeddings, with shapes (n, d) and "
                "(m, d), passed as np.ndarrays."
            )
            raise TypeError(msg)
        if not isinstance(A2, np.ndarray):
            msg = (
                f"Embedding of the second graph is of type {type(A2)}, not an "
                "array. If input_graph is False, the inputs need to be "
                "adjacency spectral embeddings, with shapes (n, d) and "
                "(m, d), passed as np.ndarrays."
            )
            raise TypeError(msg)

        if A1.ndim != 2:
            msg = (
                "Embedding array of the first graph does not have two dimensions. "
                "If input_graph is False, the inputs need to be adjacency "
                "spectral embeddings, with shapes (n, d) and (m, d)"
            )
            raise ValueError(msg)
        if A2.ndim != 2:
            msg = (
                "Embedding array of the second graph does not have two dimensions. "
                "If input_graph is False, the inputs need to be adjacency "
                "spectral embeddings, with shapes (n, d) and (m, d)"
            )
            raise ValueError(msg)
        if A1.shape[1] != A2.shape[1]:
            msg = (
                "Two input embeddings have different number of components. "
                "If input_graph is False, the inputs need to be adjacency "
                "spectral embeddings, with shapes (n, d) and (m, d)"
            )
            raise ValueError(msg)

        # checking for inf values
        X1_hat = check_array(A1)
        X2_hat = check_array(A2)

    if align_type == "sign_flips":
        aligner = SignFlips(**align_kws)
        X1_hat = aligner.fit_transform(X1_hat, X2_hat)
        Q = aligner.Q_
    elif align_type == "seedless_procrustes":
        aligner = SeedlessProcrustes(**align_kws)
        X1_hat = aligner.fit_transform(X1_hat, X2_hat)
        Q = aligner.Q_
    else:
        Q = np.identity(X1_hat.shape[0])

    if size_correction:
        X1_hat, X2_hat = _sample_modified_ase(
            X1_hat, X2_hat, workers=workers, random_state=random_state, pooled=pooled
        )

    test_obj = KSample(test, compute_distkern=metric)

    data = test_obj.test(X1_hat, X2_hat, reps=n_bootstraps, workers=workers, auto=False)

    null_distribution = test_obj.indep_test.null_dist

    misc_stats = {
        "null_distribution": null_distribution,
        "n_components": n_components,
        "Q": Q,
    }
    sample_T_statistic = data[0]
    p_value = data[1]

    return ldt_result(p_value, sample_T_statistic, misc_stats)


def _embed(A1, A2, n_components):
    if n_components is None:
        num_dims1 = select_dimension(A1)[0][-1]
        num_dims2 = select_dimension(A2)[0][-1]
        n_components = max(num_dims1, num_dims2)

    ase = AdjacencySpectralEmbed(n_components=n_components)
    X1_hat = ase.fit_transform(A1)
    X2_hat = ase.fit_transform(A2)

    if isinstance(X1_hat, tuple) and isinstance(X2_hat, tuple):
        X1_hat = np.concatenate(X1_hat, axis=-1)
        X2_hat = np.concatenate(X2_hat, axis=-1)
    elif isinstance(X1_hat, tuple) ^ isinstance(X2_hat, tuple):
        msg = (
            "input graphs do not have same directedness. "
            "consider symmetrizing the directed graph."
        )
        raise ValueError(msg)

    return X1_hat, X2_hat


def _sample_modified_ase(X, Y, workers, random_state, pooled=False):
    N, M = len(X), len(Y)

    # return if graphs are same order, else ensure X the larger graph.
    if N == M:
        return X, Y
    elif M > N:
        reverse_order = True
        X, Y = Y, X
        N, M = M, N
    else:
        reverse_order = False

    # estimate the central limit theorem variance
    if pooled:
        two_samples = np.concatenate([X, Y], axis=0)
        get_sigma = fit_plug_in_variance_estimator(two_samples)
    else:
        get_sigma = fit_plug_in_variance_estimator(X)
    X_sigmas = get_sigma(X) * (N - M) / (N * M)
    # increase the variance of X by sampling from the asy dist
    X_sampled = np.zeros(X.shape)
    rng = check_random_state(random_state)
    X_sampled = np.asarray(
        Parallel(n_jobs=workers)(
            delayed(add_variance)(X[i, :], X_sigmas[i], r)
            for i, r in zip(range(N), rng.randint(np.iinfo(np.int32).max, size=N))
        )
    )

    # return the embeddings in the appropriate order
    return (Y, X_sampled) if reverse_order else (X_sampled, Y)


def add_variance(X_orig, X_sigma, seed):
    np.random.seed(seed)
    return X_orig + stats.multivariate_normal.rvs(cov=X_sigma)
