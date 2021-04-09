# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from collections import namedtuple

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import orthogonal_procrustes

from ..align import OrthogonalProcrustes
from ..embed import AdjacencySpectralEmbed, OmnibusEmbed, select_dimension
from ..simulations import rdpg
from ..utils import import_graph, is_symmetric

lpt_result = namedtuple("lpt_result", ("p_value", "sample_T_statistic", "misc_stats"))


def latent_position_test(
    A1,
    A2,
    embedding="ase",
    n_components=None,
    test_case="rotation",
    n_bootstraps=500,
    workers=1,
):
    r"""
    Two-sample hypothesis test for the problem of determining whether two random
    dot product graphs have the same latent positions.

    This test assumes that the two input graphs are vertex aligned, that is,
    there is a known mapping between vertices in the two graphs and the input graphs
    have their vertices sorted in the same order. Currently, the function only
    supports undirected graphs.

    Read more in the `Latent Position Two-Graph Testing Tutorial
    <https://microsoft.github.io/graspologic/tutorials/inference/latent_position_test.html>`_

    Parameters
    ----------
    A1, A2 : nx.Graph, nx.DiGraph, nx.MultiDiGraph, nx.MultiGraph, np.ndarray
        The two graphs to run a hypothesis test on.
        If np.ndarray, shape must be ``(n_vertices, n_vertices)`` for both graphs,
        where ``n_vertices`` is the same for both

    embedding : string, { 'ase' (default), 'omnibus'}
        String describing the embedding method to use:

        - 'ase'
            Embed each graph separately using adjacency spectral embedding
            and use Procrustes to align the embeddings.
        - 'omnibus'
            Embed all graphs simultaneously using omnibus embedding.

    n_components : None (default), or int
        Number of embedding dimensions. If None, the optimal embedding
        dimensions are found by the Zhu and Godsi algorithm.

    test_case : string, {'rotation' (default), 'scalar-rotation', 'diagonal-rotation'}
        describes the exact form of the hypothesis to test when using 'ase' or 'lse'
        as an embedding method. Ignored if using 'omnibus'. Given two latent positions,
        :math:`X_1` and :math:`X_2`, and an orthogonal rotation matrix :math:`R` that
        minimizes :math:`||X_1 - X_2 R||_F`:

        - 'rotation'
            .. math:: H_o: X_1 = X_2 R
        - 'scalar-rotation'
            .. math:: H_o: X_1 = c X_2 R

            where :math:`c` is a scalar, :math:`c > 0`
        - 'diagonal-rotation'
            .. math:: H_o: X_1 = D X_2 R

            where :math:`D` is an arbitrary diagonal matrix

    n_bootstraps : int, optional (default 500)
        Number of bootstrap simulations to run to generate the null distribution

    workers : int (default=1)
        Number of workers to use. If more than 1, parallelizes the bootstrap simulations.
        Supply -1 to use all cores available.

    Returns
    ----------
    p_value : float
        The overall p value from the test; this is the max of 'p_value_1' and 'p_value_2'

    sample_T_statistic : float
        The observed difference between the embedded positions of the two input graphs
        after an alignment (the type of alignment depends on ``test_case``)

    misc_stats : dictionary
        A collection of other statistics obtained from the latent position test

        - 'p_value_1', 'p_value_2' : float
            The p value estimate from the null distributions from sample 1 and sample 2

        - 'null_distribution_1', 'null_distribution_2' : np.ndarray (n_bootstraps,)
            The distribution of T statistics generated under the null, using the first and
            and second input graph, respectively. The latent positions of each sample graph
            are used independently to sample random dot product graphs, so two null
            distributions are generated

    See also
    --------
    graspologic.embed.AdjacencySpectralEmbed
    graspologic.embed.OmnibusEmbed
    graspologic.embed.selectSVD

    References
    ----------
    .. [1] Tang, M., A. Athreya, D. Sussman, V. Lyzinski, Y. Park, Priebe, C.E.
       "A Semiparametric Two-Sample Hypothesis Testing Problem for Random Graphs"
       Journal of Computational and Graphical Statistics, Vol. 26(2), 2017
    """

    if type(embedding) is not str:
        raise TypeError("embedding must be str")
    if type(n_bootstraps) is not int:
        raise TypeError()
    if type(test_case) is not str:
        raise TypeError()
    if n_bootstraps < 1:
        raise ValueError(
            "{} is invalid number of bootstraps, must be greater than 1".format(
                n_bootstraps
            )
        )
    if embedding not in ["ase", "omnibus"]:
        raise ValueError("{} is not a valid embedding method.".format(embedding))
    if test_case not in ["rotation", "scalar-rotation", "diagonal-rotation"]:
        raise ValueError(
            "test_case must be one of 'rotation', 'scalar-rotation',"
            + "'diagonal-rotation'"
        )
    # check workers argument
    if not isinstance(workers, int):
        msg = "workers must be an int, not {}".format(type(workers))
        raise TypeError(msg)

    A1 = import_graph(A1)
    A2 = import_graph(A2)
    if not is_symmetric(A1) or not is_symmetric(A2):
        raise NotImplementedError()  # TODO asymmetric case
    if A1.shape != A2.shape:
        raise ValueError("Input matrices do not have matching dimensions")
    if n_components is None:
        # get the last elbow from ZG for each and take the maximum
        num_dims1 = select_dimension(A1)[0][-1]
        num_dims2 = select_dimension(A2)[0][-1]
        n_components = max(num_dims1, num_dims2)
    X_hats = _embed(A1, A2, embedding, n_components)
    sample_T_statistic = _difference_norm(X_hats[0], X_hats[1], embedding, test_case)

    # Compute null distributions
    null_distribution_1 = Parallel(n_jobs=workers)(
        delayed(_bootstrap)(X_hats[0], embedding, n_components, n_bootstraps, test_case)
        for _ in range(n_bootstraps)
    )
    null_distribution_1 = np.array(null_distribution_1)

    null_distribution_2 = Parallel(n_jobs=workers)(
        delayed(_bootstrap)(X_hats[1], embedding, n_components, n_bootstraps, test_case)
        for _ in range(n_bootstraps)
    )
    null_distribution_2 = np.array(null_distribution_2)

    # using exact mc p-values (see, for example, Phipson and Smyth, 2010)
    p_value_1 = (
        len(null_distribution_1[null_distribution_1 >= sample_T_statistic]) + 1
    ) / (n_bootstraps + 1)
    p_value_2 = (
        len(null_distribution_2[null_distribution_2 >= sample_T_statistic]) + 1
    ) / (n_bootstraps + 1)

    p_value = max(p_value_1, p_value_2)

    misc_stats = {
        "null_distribution_1": null_distribution_1,
        "null_distribution_2_": null_distribution_2,
        "p_value_1": p_value_1,
        "p_value_2": p_value_2,
    }

    return lpt_result(p_value, sample_T_statistic, misc_stats)


def _bootstrap(
    X_hat, embedding, n_components, n_bootstraps, test_case, rescale=False, loops=False
):
    A1_simulated = rdpg(X_hat, rescale=rescale, loops=loops)
    A2_simulated = rdpg(X_hat, rescale=rescale, loops=loops)
    X1_hat_simulated, X2_hat_simulated = _embed(
        A1_simulated, A2_simulated, embedding, n_components, check_lcc=False
    )
    t_bootstrap = _difference_norm(
        X1_hat_simulated, X2_hat_simulated, embedding, test_case
    )
    return t_bootstrap


def _difference_norm(X1, X2, embedding, test_case):
    if embedding in ["ase"]:
        if test_case == "rotation":
            pass
        elif test_case == "scalar-rotation":
            X1 = X1 / np.linalg.norm(X1, ord="fro")
            X2 = X2 / np.linalg.norm(X2, ord="fro")
        elif test_case == "diagonal-rotation":
            normX1 = np.sum(X1 ** 2, axis=1)
            normX2 = np.sum(X2 ** 2, axis=1)
            normX1[normX1 <= 1e-15] = 1
            normX2[normX2 <= 1e-15] = 1
            X1 = X1 / np.sqrt(normX1[:, None])
            X2 = X2 / np.sqrt(normX2[:, None])
        aligner = OrthogonalProcrustes()
        X1 = aligner.fit_transform(X1, X2)
    return np.linalg.norm(X1 - X2)


def _embed(A1, A2, embedding, n_components, check_lcc=True):
    if embedding == "ase":
        X1_hat = AdjacencySpectralEmbed(
            n_components=n_components, check_lcc=check_lcc
        ).fit_transform(A1)
        X2_hat = AdjacencySpectralEmbed(
            n_components=n_components, check_lcc=check_lcc
        ).fit_transform(A2)
    elif embedding == "omnibus":
        X_hat_compound = OmnibusEmbed(
            n_components=n_components, check_lcc=check_lcc
        ).fit_transform((A1, A2))
        X1_hat = X_hat_compound[0]
        X2_hat = X_hat_compound[1]
    return (X1_hat, X2_hat)
