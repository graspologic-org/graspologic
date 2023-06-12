import warnings
from collections import namedtuple
from typing import Union

import numpy as np
import pandas as pd
from beartype import beartype
from scipy.stats import combine_pvalues
from statsmodels.stats.multitest import multipletests

from ..types import AdjacencyMatrix, List
from ..utils import import_graph, is_loopless, is_symmetric, is_unweighted, remove_loops
from .binomial import BinomialTestMethod, binom_2samp
from .utils import compute_density

Labels = Union[np.ndarray, List]

SBMResult = namedtuple(
    "SBMResult", ["probabilities", "observed", "possible", "group_counts"]
)

GroupTestResult = namedtuple("GroupTestResult", ["stat", "pvalue", "misc"])


def fit_sbm(A: AdjacencyMatrix, labels: Labels, loops: bool = False) -> SBMResult:
    """
    Fits a stochastic block model to data for a given network with known group
    identities. Required inputs are the adjacency matrix for the
    network and the group label for each node of the network. The number of labels must
    equal the number of nodes of the network.

    For each possible group-to-group connection, e.g. group 1-to-group 1,
    group 1-to-group 2, etc., this function computes the total number
    of possible edges between the groups, the actual number of edges connecting the
    groups, and the estimated probability of an edge
    connecting each pair of groups (i.e. B_hat). The function also calculates and
    returns the total number of nodes corresponding to each
    provided label.

    Parameters
    ----------
    A: np.array, int shape(n1,n1)
        The adjacency matrix for the network at issue. Entries are either 1
        (edge present) or 0 (edge absent). This is a square matrix with
        side length equal to the number of nodes in the network.

    labels: array-like, int shape(n1,1)
        The group labels corresponding to each node in the network. This is a
        one-dimensional array with a number of entries equal to the
        number of nodes in the network.

    loops: boolean
        This parameter instructs the function to either include or exclude self-loops
        (i.e. connections between a node and itself) when
        fitting the SBM model. This parameter is optional; default is false, meaning
        self-loops will be excluded.

    Returns
    -------
    SBMResult: namedtuple
        This function returns a namedtuple with four key/value pairs:
            ("probabilities",B_hat):
                B_hat: array-like, float shape((number of unique labels)^2,1)
                    This variable stores the computed edge probabilities for all
                    possible group-to-group connections, computed as the ratio
                    of number of actual edges to number of possible edges.
            ("observed",n_observed):
                n_observed: dataframe
                    This variable stores the number of observed edges for each
                    group-to-group connection. Data is indexed as the number
                    of edges between each source group and each target group.
            ("possibe",n_possible):
                n_possible: dataframe
                    This variable stores the total number of possible edges for each
                    group-to-group connection. Indexing is identical to
                    the above. Network density for each group-to-group connection can
                    easily be determined as n_observed/n_possible.
            ("group_counts",counts_labels):
                counts_labels: pd.series
                    This variable stores the number of nodes belonging to each group
                    label.

    """

    if not loops:
        A = remove_loops(A)

    n = A.shape[0]

    node_to_comm_map = dict(zip(np.arange(n), labels))

    # map edges to their incident communities
    source_inds, target_inds = np.nonzero(A)
    comm_mapper = np.vectorize(node_to_comm_map.get)
    source_comm = comm_mapper(source_inds)
    target_comm = comm_mapper(target_inds)

    # get the total number of possible edges for each community -> community cell
    unique_labels, counts_labels = np.unique(labels, return_counts=True)
    K = len(unique_labels)

    n_observed = (
        pd.crosstab(
            source_comm,
            target_comm,
            dropna=False,
            rownames=["source"],
            colnames=["target"],
        )
        .reindex(index=unique_labels, columns=unique_labels)
        .fillna(0.0)
    )

    num_possible = np.outer(counts_labels, counts_labels)

    if not loops:
        # then there would have been n fewer possible edges
        num_possible[np.arange(K), np.arange(K)] = (
            num_possible[np.arange(K), np.arange(K)] - counts_labels
        )

    n_possible = pd.DataFrame(
        data=num_possible, index=unique_labels, columns=unique_labels
    )
    n_possible.index.name = "source"
    n_possible.columns.name = "target"

    B_hat = np.divide(n_observed, n_possible)

    counts_labels = pd.Series(index=unique_labels, data=counts_labels)

    return SBMResult(B_hat, n_observed, n_possible, counts_labels)


def _make_adjacency_dataframe(data: AdjacencyMatrix, index: Labels) -> pd.DataFrame:
    """
    Helper function to convert data with a given index into a dataframe data structure.
    """
    df = pd.DataFrame(data=data, index=index.copy(), columns=index.copy())
    df.index.name = "source"
    df.columns.name = "target"
    return df


@beartype
def group_connection_test(
    A1: AdjacencyMatrix,
    A2: AdjacencyMatrix,
    labels1: Labels,
    labels2: Labels,
    density_adjustment: Union[bool, float] = False,
    method: BinomialTestMethod = "score",
    combine_method: str = "tippett",
    correct_method: str = "bonferroni",
    alpha: float = 0.05,
) -> GroupTestResult:
    r"""
    Compares two networks by testing whether edge probabilities between groups are
    significantly different for the two networks under a stochastic block model
    assumption.

    This function requires the group labels in both networks to be known and to have the
    same categories; although the exact number of nodes belonging to each group does not
    need to be identical. Note that using group labels inferred from the data may
    yield an invalid test.

    This function also permits the user to test whether one network's group connection
    probabilities are a constant multiple of the other's (see ``density_adjustment``
    parameter).

    Parameters
    ----------
    A1: np.array, shape(n1,n1)
        The adjacency matrix for network 1. Will be treated as a binary network,
        regardless of whether it was weighted.
    A2 np.array, shape(n2,n2)
        The adjacency matrix for network 2. Will be treated as a binary network,
        regardless of whether it was weighted.
    labels1: array-like, shape (n1,)
        The group labels for each node in network 1.
    labels2: array-like, shape (n2,)
        The group labels for each node in network 2.
    density_adjustment: boolean, optional
        Whether to perform a density adjustment procedure. If ``True``, will test the
        null hypothesis that the group-to-group connection probabilities of one network
        are a constant multiple of those of the other network. Otherwise, no density
        adjustment will be performed.
    method: str, optional
        Specifies the statistical test to be performed to compare each of the
        group-to-group connection probabilities. By default, this performs
        the score test (essentially equivalent to chi-squared test when
        ``density_adjustment=False``), but the user may also enter "chi2" to perform the
        chi-squared test, or "fisher" for Fisher's exact test.
    combine_method: str, optional
        Specifies the method for combining p-values (see Notes and [1]_ for more
        details). Default is "tippett" for Tippett's method (recommended), but the user
        can also enter any other method supported by
        :func:`scipy.stats.combine_pvalues`.
    correct_method: str, optional
        Specifies the method for correcting for multiple comparisons. Default value is
        "holm" to use the Holm-Bonferroni correction method, but
        many others are possible (see :func:`statsmodels.stats.multitest.multipletests`
        for more details and options).
    alpha: float, optional
        The significance threshold. By default, this is the conventional value of
        0.05 but any value on the interval :math:`[0,1]` can be entered. This only
        affects the results in ``misc['rejections']``.

    Returns
    -------
    GroupTestResult: namedtuple
        A tuple containing the following data:

        stat: float
            The statistic computed by the method chosen for combining
            p-values (see ``combine_method``).
        pvalue: float
            The p-value for the overall network-to-network comparison using under a
            stochastic block model assumption. Note that this is the p-value for the
            comparison of the entire group-to-group connection matrices
            (i.e., :math:`B_1` and :math:`B_2`).
        misc: dict
            A dictionary containing a number of statistics relating to the individual
            group-to-group connection comparisons.

                "uncorrected_pvalues", pd.DataFrame
                    The p-values for each group-to-group connection comparison, before
                    correction for multiple comparisons.
                "stats", pd.DataFrame
                    The test statistics for each of the group-to-group comparisons,
                    depending on ``method``.
                "probabilities1", pd.DataFrame
                    This contains the B_hat values computed in fit_sbm above for
                    network 1, i.e. the hypothesized group connection density for
                    each group-to-group connection for network 1.
                "probabilities2", pd.DataFrame
                    Same as above, but for network 2.
                "observed1", pd.DataFrame
                    The total number of observed group-to-group edge connections for
                    network 1.
                "observed2", pd.DataFrame
                    Same as above, but for network 2.
                "possible1", pd.DataFrame
                    The total number of possible edges for each group-to-group pair in
                    network 1.
                "possible2", pd.DataFrame
                    Same as above, but for network 2.
                "group_counts1", pd.Series
                    Contains total number of nodes corresponding to each group label for
                    network 1.
                "group_counts2", pd.Series
                    Same as above, for network 2
                "null_ratio", float
                    If the "density adjustment" parameter is set to "true", this
                    variable contains the null hypothesis for the quotient of
                    odds ratios for the group-to-group connection densities for the two
                    networks. In other words, it contains the hypothesized
                    factor by which network 1 is "more dense" or "less dense" than
                    network 2. If "density adjustment" is set to "false", this
                    simply returns a value of 1.0.
                "n_tests", int
                    This variable contains the number of group-to-group comparisons
                    performed by the function.
                "rejections", pd.DataFrame
                    Contains a square matrix of boolean variables. The side length of
                    the matrix is equal to the number of distinct group
                    labels. An entry in the matrix is "true" if the null hypothesis,
                    i.e. that the group-to-group connection density
                    corresponding to the row and column of the matrix is equal for both
                    networks (with or without a density adjustment factor),
                    is rejected. In simpler terms, an entry is only "true" if the
                    group-to-group density is statistically different between
                    the two networks for the connection from the group corresponding to
                    the row of the matrix to the group corresponding to the
                    column of the matrix.
                "corrected_pvalues", pd.DataFrame
                    Contains the p-values for the group-to-group connection densities
                    after correction using the chosen correction_method.

    Notes
    -----
    Under a stochastic block model assumption, the probability of observing an edge from
    any node in group :math:`i` to any node in group :math:`j` is given by
    :math:`B_{ij}`, where :math:`B` is a :math:`K \times K` matrix of connection
    probabilities if there are :math:`K` groups. This test assumes that both networks
    came from a stochastic block model with the same number of groups, and a fixed
    assignment of nodes to groups. The null hypothesis is that the group-to-group
    connection probabilities are the same

    .. math:: H_0: B_1 = B_2

    The alternative hypothesis is that they are not the same

    .. math:: H_A: B_1 \neq B_2

    Note that this alternative includes the case where even just one of these
    group-to-group connection probabilities are different between the two networks. The
    test is conducted by first comparing each group-to-group connection via its own
    test, i.e.,

    .. math:: H_0: {B_{1}}_{ij} = {B_{2}}_{ij}

    .. math:: H_A: {B_{1}}_{ij} \neq {B_{2}}_{ij}

    The p-values for each of these individual comparisons are stored in
    ``misc['uncorrected_pvalues']``, and after multiple comparisons correction, in
    ``misc['corrected_pvalues']``. The test statistic and p-value returned by this
    test are for the overall comparison of the entire group-to-group connection
    matrices. These are computed by appropriately combining the p-values for each of
    the individual comparisons. For more details, see [1]_.

    When ``density_adjustment`` is set to ``True``, the null hypothesis is adjusted to
    account for the fact that the group-to-group connection densities may be different
    only up to a multiplicative factor which sets the densities of the two networks
    the same in expectation. In other words, the null and alternative hypotheses are
    adjusted to be

    .. math:: H_0: B_1 = c B_2

    .. math:: H_A: B_1 \neq c B_2

    where :math:`c` is a constant which sets the densities of the two networks the same.

    Note that in cases where one of the networks has no edges in a particular
    group-to-group connection, it is nonsensical to run a statistical test for that
    particular connection. In these cases, the p-values for that individual comparison
    are set to ``np.nan``, and that test is not included in the overall test statistic
    or multiple comparison correction.

    This test makes several assumptions about the data and test (which could easily be
    loosened in future versions):

            We assume that the networks are directed. If the networks are undirected
            (and the adjacency matrices are thus symmetric), then edges would be counted
            twice, which would lead to an incorrect calculation of the edge probability.
            We believe passing in the upper or lower triangle of the adjacency matrix
            would solve this, but this has not been tested.

            We assume that the networks are loopless, that is we do not consider the
            probability of an edge existing between a node and itself. This can be
            weakened and made an option in future versions.

            We only implement the alternative hypothesis of "not equals" (two-sided);
            future versions could implement the one-sided alternative hypotheses.

    References
    ----------
    .. [1] Pedigo, B.D., Powell, M., Bridgeford, E.W., Winding, M., Priebe, C.E.,
           Vogelstein, J.T. "Generative network modeling reveals quantitative
           definitions of bilateral symmetry exhibited by a whole insect brain
           connectome," eLife (2023): e83739.
    """

    A1 = import_graph(A1)
    A2 = import_graph(A2)

    if is_symmetric(A1) or is_symmetric(A2):
        msg = (
            "This test assumes that the networks are directed, "
            "but one or both adjacency matrices are symmetric."
        )
        warnings.warn(msg)
    if (not is_unweighted(A1)) or (not is_unweighted(A2)):
        msg = (
            "This test assumes that the networks are unweighted, "
            "but one or both adjacency matrices are weighted."
            "Test will be run on the binarized version of these adjacency matrices."
        )
        warnings.warn(msg)
    if (not is_loopless(A1)) or (not is_loopless(A2)):
        msg = (
            "This test assumes that the networks are loopless, "
            "but one or both adjacency matrices have self-loops."
            "Test will be run on the loopless version of these adjacency matrices."
        )
        warnings.warn(msg)

    B1, n_observed1, n_possible1, group_counts1 = fit_sbm(A1, labels1)
    B2, n_observed2, n_possible2, group_counts2 = fit_sbm(A2, labels2)
    if not n_observed1.index.equals(n_observed2.index):
        raise ValueError()
    elif not n_observed1.columns.equals(n_observed2.columns):
        raise ValueError()
    elif not n_possible1.index.equals(n_possible2.index):
        raise ValueError()
    elif not n_observed1.columns.equals(n_observed2.columns):
        raise ValueError()

    index = n_observed1.index.copy()

    if n_observed1.shape[0] != n_observed2.shape[0]:
        raise ValueError()

    K = n_observed1.shape[0]

    uncorrected_pvalues_temp = np.empty(
        (K, K), dtype=float
    )  # had to make a new variable to keep mypy happy
    uncorrected_pvalues = _make_adjacency_dataframe(uncorrected_pvalues_temp, index)

    stats_temp = np.empty((K, K), dtype=float)
    stats = _make_adjacency_dataframe(stats_temp, index)

    if density_adjustment != False:  # cause could be float
        if density_adjustment == True:
            density1 = compute_density(A1)
            density2 = compute_density(A2)
            adjustment_factor = density1 / density2
        else:
            adjustment_factor = density_adjustment
    else:
        adjustment_factor = 1.0

    for i in index:
        for j in index:
            curr_stat, curr_pvalue = binom_2samp(
                n_observed1.loc[i, j],
                n_possible1.loc[i, j],
                n_observed2.loc[i, j],
                n_possible2.loc[i, j],
                method=method,
                null_ratio=adjustment_factor,
            )
            uncorrected_pvalues.loc[i, j] = curr_pvalue
            stats.loc[i, j] = curr_stat

    misc = {}
    misc["uncorrected_pvalues"] = uncorrected_pvalues
    misc["stats"] = stats
    misc["probabilities1"] = B1
    misc["probabilities2"] = B2
    misc["observed1"] = n_observed1
    misc["observed2"] = n_observed2
    misc["possible1"] = n_possible1
    misc["possible2"] = n_possible2
    misc["group_counts1"] = group_counts1
    misc["group_counts2"] = group_counts2
    misc["null_ratio"] = adjustment_factor

    run_pvalues = uncorrected_pvalues.values
    indices = np.nonzero(~np.isnan(run_pvalues))
    run_pvalues = run_pvalues[indices]
    n_tests = len(run_pvalues)
    misc["n_tests"] = n_tests

    # correct for multiple comparisons
    rejections_flat, corrected_pvalues_flat, _, _ = multipletests(
        run_pvalues,
        alpha,
        method=correct_method,
        is_sorted=False,
        returnsorted=False,
    )
    rejections = np.full((K, K), False, dtype=bool)
    rejections[indices] = rejections_flat
    rejections = _make_adjacency_dataframe(rejections, index)
    misc["rejections"] = rejections

    corrected_pvalues = np.full((K, K), np.nan, dtype=float)
    corrected_pvalues[indices] = corrected_pvalues_flat
    corrected_pvalues = _make_adjacency_dataframe(corrected_pvalues, index)
    misc["corrected_pvalues"] = corrected_pvalues

    # combine p-values (on the UNcorrected p-values)
    if run_pvalues.min() == 0.0:
        # TODO consider raising a new warning here
        stat = np.inf
        pvalue = 0.0
    else:
        stat, pvalue = combine_pvalues(run_pvalues, method=combine_method)
    return GroupTestResult(stat, pvalue, misc)
