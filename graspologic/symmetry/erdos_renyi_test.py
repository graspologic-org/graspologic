import numpy as np

from .group_connection_test import group_connection_test, group_connection_test_paired


def _squeeze_value(old_misc, new_misc, old_key, new_key):

    """
    Helper function to rename the keys for a dictionary variable. Takes the old and new dictionaries, and the old and new keys, as
    arguments, and returns the new dictionary, which uses the new keys to index the data.
    """
    variable = old_misc[old_key]
    variable = variable.values[0, 0]
    new_misc[new_key] = variable


def erdos_renyi_test(A1, A2, method="fisher"):

    """
    This function performs the Erdos-Renyi test to compare the adjacency matrices for two networks. Under the Erdos-Renyi model, it is
    assumed that the probability of an edge existing between any two adjacent nodes is equal to some constant p. This function tests
    whether the probability of an edge in network 1 is statistically different from that in network 2. In other words, it tests a null
    hypothesis that p1, the edge probability for network 1, is equal to p2, the edge probability for network 2.

    The Erdos-Renyi model is essentially a special case of stochastic block model where all nodes belong to the same group. Thus, to
    perform the required calculations, this function calls group_connection_test, but assigns all nodes to a single group. For further
    information regarding the group_connection_test function, consult the documentation for the .sbm file in this package.

    Parameters
    ------------
    A1: np. array, int
        The adjacency matrix for network 1. Contains either a 0 or 1 at each location in the array, where a 1 denotes an edge and a 0 denotes
        the absence of an edge.
    A2: np. array, int
        Adjacency matrix for network 2.
    method: string, optional, default="fisher"
        Specifies the statistical test to be performed to reject or fail to reject the null hypothesis. The default option is "fisher",
        which uses Fisher's exact test, but the user may also enter "chi2" to use a chi-squared test. Any other entry will give an error.

    Returns
    --------
    stat: float
        This returns a statistic calculated by group_connection_test when combining p-values for multiple group-to-group comparisons. This
        won't be too meaningful or useful for the Erdos-Renyi test.
    pvalue: float
        The computed probability that the null hypothesis (i.e. p1=p2) is correct given the observed pair of networks. A value below 0.05
        is required by convention in order to deduce statistical significance.
    er_misc: dict
        Dictionary containing a number of computed statistics for the network comparison performed:
            "probability1" = float
                This contains the computed probability of an edge between nodes in network 1. In other words, this is p1
            "probabilities2" = float
                This contains p2, i.e. the computed network density of network 2.
            "observed1" = n_observed1, dataframe
                The total number of edge connections for network 1.
            "observe2" = n_observed2, dataframe
                Same as above, but for network 2.
            "possible1" = n_possible1, dataframe
                The total number of possible edges for network 1.
            "possible2" = n_possible2, dataframe
                Same as above, but for network 2.

    """
    stat, pvalue, sbm_misc = group_connection_test(
        A1,
        A2,
        labels1=np.ones(A1.shape[0]),
        labels2=np.ones(A2.shape[0]),
        method=method,
    )
    old_keys = [
        "probabilities1",
        "probabilities2",
        "observed1",
        "observed2",
        "possible1",
        "possible2",
    ]
    new_keys = [
        "probability1",
        "probability2",
        "observed1",
        "observed2",
        "possible1",
        "possible2",
    ]
    er_misc = {}
    for old_key, new_key in zip(old_keys, new_keys):
        _squeeze_value(sbm_misc, er_misc, old_key, new_key)

    return stat, pvalue, er_misc


def erdos_renyi_test_paired(A1, A2):
    """
    Performs the Erdos-Renyi test on paired data, i.e. where there is a known node-to-node correspondence between the two networks.

    Parameters:
    -----------
    A1: np.array, int
        The adjacency matrix for network 1.
    A2: np. array, int
        Same, but for network 2.

    Returns:
    --------
    stat: float
        This contains the a statistic computed by the algorithm for comining p-values. Again, not too useful for the Erdos-Renyi test.
    pvalue: float
        The combined p-value for the total network-to-network comparison using the Erdos-Renyi model, calculated using Tippett's method.
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
    """
    stat, pvalue, sbm_misc = group_connection_test_paired(
        A1, A2, labels=np.ones(A1.shape[0])
    )
    misc = {}
    misc["both"] = sbm_misc["both"].loc[1, 1]
    misc["neither"] = sbm_misc["neither"].loc[1, 1]
    misc["only1"] = sbm_misc["only1"].loc[1, 1]
    misc["only2"] = sbm_misc["only2"].loc[1, 1]
    return stat, pvalue, misc
