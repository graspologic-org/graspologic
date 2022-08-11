from collections import namedtuple

import numpy as np
import pandas as pd
from graspologic.utils import remove_loops
from statsmodels.stats.multitest import multipletests

from .binomial import binom_2samp, binom_2samp_paired
from .combine import combine_pvalues
from .utils import compute_density_adjustment
from beartype import beartype

SBMResult = namedtuple(
    "sbm_result", ["probabilities", "observed", "possible", "group_counts"]
)


def fit_sbm(A, labels, loops=False):

    """
    Fits a stochastic block model to data for a given network with known group identities. Required inputs are the adjacency matrix for the
    network and the group label for each node of the network. The number of labels must equal the number of nodes of the network.

    For each possible group-to-group connection, e.g. group 1-to-group 1, group 1-to-group 2, etc., this function computes the total number
    of possible edges between the groups, the actual number of edges connecting the groups, and the estimated probability of an edge
    connecting each pair of groups (i.e. B_hat). The function also calculates and returns the total number of nodes corresponding to each
    provided label.

    Parameters
    ----------
    A: np.array, int shape(num_nodes,num_nodes)
        The adjacency matrix for the network at issue. Entries are either 1 (edge present) or 0 (edge absent). This is a square matrix with
        side length equal to the number of nodes in the network.

    labels: array-like, int shape(num_nodes,1)
        The group labels corresponding to each node in the network. This is a one-dimensional array with a number of entries equal to the
        number of nodes in the network.

    loops: boolean
        This parameter instructs the function to either include or exclude self-loops (i.e. connections between a node and itself) when
        fitting the SBM model. This parameter is optional; default is false, meaning self-loops will be excluded.

    Returns
    -------
    SBMResult: namedtuple
        This function returns a namedtuple with four key/value pairs:
            ("probabilities",B_hat):
                B_hat: array-like, float shape((number of unique labels)^2,1)
                    This variable stores the computed edge probabilities for all possible group-to-group connections, computed as the ratio
                    of number of actual edges to number of possible edges.
            ("observed",n_observed):
                n_observed: dataframe
                    This variable stores the number of observed edges for each group-to-group connection. Data is indexed as the number
                    of edges between each source group and each target group.
            ("possibe",n_possible):
                n_possible: dataframe
                    This variable stores the total number of possible edges for each group-to-group connection. Indexing is identical to
                    the above. Network density for each group-to-group connection can easily be determined as n_observed/n_possible.
            ("group_counts",counts_labels):
                counts_labels: pd.series
                    This variable stores the number of nodes belonging to each group label.

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

    n_possible = np.outer(counts_labels, counts_labels)

    if not loops:
        # then there would have been n fewer possible edges
        n_possible[np.arange(K), np.arange(K)] = (
            n_possible[np.arange(K), np.arange(K)] - counts_labels
        )

    n_possible = pd.DataFrame(
        data=n_possible, index=unique_labels, columns=unique_labels
    )
    n_possible.index.name = "source"
    n_possible.columns.name = "target"

    B_hat = np.divide(n_observed, n_possible)

    counts_labels = pd.Series(index=unique_labels, data=counts_labels)

    return SBMResult(B_hat, n_observed, n_possible, counts_labels)


def _make_adjacency_dataframe(data, index):
    """
    Helper function to convert data with a given index into a dataframe data structure.
    """
    df = pd.DataFrame(data=data, index=index.copy(), columns=index.copy())
    df.index.name = "source"
    df.columns.name = "target"
    return df


def group_connection_test(
    A1,
    A2,
    labels1,
    labels2,
    density_adjustment=False,
    method="fisher",
    combine_method="tippett",
    correct_method="bonferroni",
    alpha=0.05,
):

    """
    Compares two sets of group-to-group connection data for two networks, to assess whether the data are statistically different. To do this,
    the function first compares each individual group-to-group connection to determine whether there is a statistically significant
    difference in the connection densities for each group-to-group pair. This is accomplished using Fisher's exact test (or another test
    selected by the user) to compare the number of observed edges versus the number of possible edges in each network. Once p values for all
    group-to-group connections are determined, the p-values are combined into a single p-value that encapsulates whether the two networks,
    as a whole, are statistically different. This procedure is described in greater detail in [#BEN'S PAPER].

    This function requires the group labels in both networks to be known and identical, although the exact number of nodes belonging to each
    group does not need to be identical.

    This function also permits the user to test whether one network is a fixed multiple more dense or less dense than the other network. This
    procedure is referred to as the "density-adjusted group connection test" in [#BEN'S PAPER]. To do this, the user simply includes the
    argument density_adjustment=True. The function will then automatically compute the hypothesized density ratio for the two networks and
    determines whether to reject or fail to reject to hypothesis that one network's density is a fixed multiple of the other network's density.


    Parameters
    -----------
    A1: np.array, int shape(num_nodes,num_nodes)
        The adjacency matrix for the first network at issue. Entries are either 1 (edge present) or 0 (edge absent). This is a square
        matrix with side length equal to the number of nodes in the network.
    A2 np.array, int shape(num_nodes,num_nodes)
        The adjacency matrix for the second network at issue. Same properties as above.
    labels1: array-like, int shape(num_nodes,1)
        This variable contains the group labels for each node in network 1.
    labels2: array-like, int shape(num_nodes,1)
        This variable contains the group labels for each node in network 2.
    density_adjustment: boolean, optional
        This variable instructs the function whether to perform the density adjustment procedure alluded to above. If this variable is set
        to "true", the function will test the null hypothesis that the group-to-group connection density of one network is a fixed multiple
        of the density of that of the other network. If the variable is set to "false", which is the default setting, no density adjustment
        will be perform and the function will test the null hypothesis that the two networks have equal group-to-group connection densities.
    method: str, optional
        Specifies the statistical test to be performed to compare the group-to-group connection densities. By default, this performs
        Fisher's exact test, but the user may also enter "chi2" to perform the chi-squared test. Any entry other than "fisher" or "chi2"
        will raise an error.
    combine_method: str, optional
        Specifies the statistical method for combining p-values. Default is "tippett" for Tippett's method, but the user can also enter
        "fisher" to use Fisher's method. Any other entry will cause an error.
    correct_method: str, optional
        Specifies the statistical method for correcting for multiple comparisons. Since this function is performing many comparisons
        between subsets of the data, the probability of observing a "statistically significant" result by pure chance is increased. A
        correction is performed to adjust for this phenomenon. Default value is "bonferroni" to use Bonferroni correction method, but
        many others are possible (see https://www.statsmodels.org/dev/generated/statsmodels.stats.multitest.multipletests.html)
    alpha: float, optional
        The value to be used when testing the statistical significance of the results. By default, this is the conventional value of 0.05
        but any value on the interval [0,1] can be entered.

    Returns
    -------
    stat: float
        This contains the a statistic computed by the method chosen for combining p values (i.e. "combine_method"). For Tippett's method,
        this is the least of the p values. For Fisher's method, this is the test statistic computed as -2*sum(log(p-values)).
    pvalue: float
        The combined p-value for the total network-to-network comparison using the SBM model, calculated using the chosen combine_method.
    misc: dict
        A dictionary containing a number of statistics relating to the individual group-to-group connection comparisons.
            "uncorrected_pvalues" = uncorrected_pvalues, array-like, float
                The p-values for each group-to-group connection comparison, before correction for multiple comparisons.
            "stats" = stats, array-like, float.
                The odds ratio for the provided data, representing the prior probability of a "success" (in this
                case, the odds of an edge occurring between two nodes).
            "probabilities1" = B1, array-like, float
                This contains the B_hat values computed in fit_sbm above for network 1, i.e. the hypothesized group connection density for
                each group-to-group connection for network 1.
            "probabilities2" = B2, array-like, float
                Same as above, but for network 2.
            "observed1" = n_observed1, dataframe
                The total number of observed group-to-group edge connections for network 1.
            "observed2" = n_observed2, dataframe
                Same as above, but for network 2.
            "possible1" = n_possible1, dataframe
                The total number of possible edges for each group-to-group pair in network 1.
            "possible2" = n_possible2, dataframe
                Same as above, but for network 2.
            "group_counts1" = group_counts1, pd.series
                Contains total number of nodes corresponding to each group label for network 1.
            "group_counts2" = group_counts2, pd.series
                Same as above, for network 2
            "null_ratio" = adjustment_factor, float
                If the "density adjustment" parameter is set to "true", this variable contains the null hypothesis for the quotient of
                odds ratios for the group-to-group connection densities for the two networks. In other words, it contains the hypothesized
                factor by which network 1 is "more dense" or "less dense" than network 2. If "density adjustment" is set to "false", this
                simply returns a value of 1.0.
            "n_tests" = n_tests, integer
                This variable contains the number of group-to-group comparisons performed by the function.
            "rejections" = rejections, dataframe
                Contains a square matrix of boolean variables. The side length of the matrix is equal to the number of distinct group
                labels. An entry in the matrix is "true" if the null hypothesis, i.e. that the group-to-group connection density
                corresponding to the row and column of the matrix is equal for both networks (with or without a density adjustment factor),
                is rejected. In simpler terms, an entry is only "true" if the group-to-group density is statistically different between
                the two networks for the connection from the group corresponding to the row of the matrix to the group corresponding to the
                column of the matrix.
            "corrected_pvalues" = corrected_pvalues, dataframe
                Contains the p-values for the group-to-group connection densities after correction using the chosen correction_method.


    Notes
    ------
    Default value of "correct_method" in the bilateral-connectome repository is "holm". I changed this to "bonferroni" because that is
    the method chosen in the paper.

    The function name has been changed to group_connection_test to match the chosen language in the paper.
    """

    B1, n_observed1, n_possible1, group_counts1 = fit_sbm(A1, labels1)
    B2, n_observed2, n_possible2, group_counts2 = fit_sbm(A2, labels2)

    # TODO fix this up
    assert n_observed1.index.equals(n_observed2.index)
    assert n_observed1.columns.equals(n_observed2.columns)
    assert n_possible1.index.equals(n_possible2.index)
    assert n_observed1.columns.equals(n_possible2.columns)
    index = n_observed1.index.copy()

    if n_observed1.shape[0] != n_observed2.shape[0]:
        raise ValueError()

    K = n_observed1.shape[0]

    uncorrected_pvalues = np.empty((K, K), dtype=float)
    uncorrected_pvalues = _make_adjacency_dataframe(uncorrected_pvalues, index)

    stats = np.empty((K, K), dtype=float)
    stats = _make_adjacency_dataframe(stats, index)

    if density_adjustment:
        adjustment_factor = compute_density_adjustment(A1, A2)
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
    return stat, pvalue, misc


def offdiag_indices_from(arr):
    """
    Helper function for isolating the off-diagonal indices from a matrix.
    """
    upper_rows, upper_cols = np.triu_indices_from(arr, k=1)
    lower_rows, lower_cols = np.tril_indices_from(arr, k=-1)
    rows = np.concatenate((upper_rows, lower_rows))
    cols = np.concatenate((upper_cols, lower_cols))
    return rows, cols


def group_connection_test_paired(
    A1,
    A2,
    labels,
):
    """
    Very similar to the above, except this function performs a paired version of the group connection test, meaning that there is a node-
    to-node correspondence between the two networks. This function currently only determines the p-value for the overall group-to-group
    comparison, and does not compute the individual, corrected group-to-group p-values. This function uses Tippett's method for combining
    p-values.

    Parameters
    -----------
    A1: np.array, int
        The adjacency matrix for the first network. This is a square matrix with side length corresponding to the number of nodes in the
        network. Entries in the matrix are either 1, denoting an edge, or 0, denoting the absence of an edge.
    A2: np.array, int
        The adjacency matrix for the second network.
    labels: array-like, int
        The group labels for the nodes in the networks (which are presumed to be identical for both networks.)

    Returns
    --------
    stat: float
        This contains the a statistic computed by the algorithm for comining p-values. For Tippett's method,
        this is the least of the p values.
    pvalue: float
        The combined p-value for the total network-to-network comparison using the SBM model, calculated using Tippett's method.
    misc: dict
        Dictionary containing a few miscellaneous statistics computed by the function.
            "both" = n_both, int
                The number of locations in the adjacency matrix where a "1" appears in both A1 and A2, i.e. the number of locations where
                an edge appears in both network 1 and network 2.
            "only1" = only1, int
                The number of locations where an edge appears in network 1 but not network 2.
            "only2" = only2, int
                The number of locations where an edge appears in network 2 but not network 1.
            "neither" = neither
                The number of locations where there is no edge in either network.
            "uncorrected_pvalues" = uncorrected_pvalues, dataframe
                A matrix containing the uncorrected p-values for each group-to-group comparison.
            "n_tests" = n_tests, int
                The number of group-to-group comparisons performed.
    """
    index, group_indices, group_counts = np.unique(
        labels, return_counts=True, return_inverse=True
    )

    if A1.shape != A2.shape:
        raise ValueError("Matrices must be of equal size to use paired test")
    K = len(index)

    empty = np.empty((K, K), dtype=float)
    uncorrected_pvalues = _make_adjacency_dataframe(empty.copy(), index)
    stats = _make_adjacency_dataframe(empty.copy(), index)
    empty = np.empty((K, K), dtype=int)
    both = _make_adjacency_dataframe(empty.copy(), index)
    neither = _make_adjacency_dataframe(empty.copy(), index)
    only1 = _make_adjacency_dataframe(empty.copy(), index)
    only2 = _make_adjacency_dataframe(empty.copy(), index)

    for i, source_group in enumerate(index):
        source_mask = group_indices == i
        for j, target_group in enumerate(index):
            target_mask = group_indices == j
            A1_subgraph = A1[source_mask][:, target_mask]
            A2_subgraph = A2[source_mask][:, target_mask]

            if i == j:
                rows, cols = offdiag_indices_from(A1_subgraph)

                edges1 = A1_subgraph[rows, cols]
                edges2 = A2_subgraph[rows, cols]
            else:
                r1, c1 = A1_subgraph.shape
                r2, c2 = A2_subgraph.shape

                edges1 = A1_subgraph.reshape((r1 * c1, 1))
                edges2 = A2_subgraph.reshape((r2 * c2, 1))

            curr_stat, curr_pvalue, curr_misc = binom_2samp_paired(edges1, edges2)

            stats.loc[source_group, target_group] = curr_stat
            uncorrected_pvalues.loc[source_group, target_group] = curr_pvalue
            both.loc[source_group, target_group] = curr_misc["n_both"]
            neither.loc[source_group, target_group] = curr_misc["n_neither"]
            only1.loc[source_group, target_group] = curr_misc["n_only1"]
            only2.loc[source_group, target_group] = curr_misc["n_only2"]

    misc = {}
    misc["both"] = both
    misc["neither"] = neither
    misc["only1"] = only1
    misc["only2"] = only2
    misc["uncorrected_pvalues"] = uncorrected_pvalues

    run_pvalues = uncorrected_pvalues.values
    run_pvalues = run_pvalues[~np.isnan(run_pvalues)]
    stat, pvalue = combine_pvalues(run_pvalues, method="tippett")
    n_tests = len(run_pvalues)
    misc["n_tests"] = n_tests
    return stat, pvalue, misc
