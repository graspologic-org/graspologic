from collections import namedtuple

import numpy as np

from ..types import AdjacencyMatrix
from .group_connection_test import group_connection_test
from beartype import beartype

DensityTestResult = namedtuple("DensityTestResult", ["stat", "pvalue", "misc"])


@beartype
def density_test(
    A1: AdjacencyMatrix, A2: AdjacencyMatrix, method: str = "fisher"
) -> DensityTestResult:

    """
    This function uses the Erdos-Renyi model to perform a density test to compare the
    adjacency matrices for two networks. Under the Erdos-Renyi model, it is assumed that
    the probability of an edge existing between any two nodes is some constant, p.
    This function tests whether the probability of an edge in network 1 is
    statistically different from that in network 2. In other words, it tests a null
    hypothesis that p1, the edge probability for network 1, is equal to p2, the edge
    probability for network 2.

    Parameters
    ----------
    A1: np. array, int
        The adjacency matrix for network 1. Will be treated as a binary network,
        regardless of whether it was weighted.
    A2: np. array, int
        Adjacency matrix for network 2. Will be treated as a binary network,
        regardless of whether it was weighted.
    method: string, optional, default="fisher"
        Specifies the statistical test to be used. The default option is "fisher",
        which uses Fisher's exact test, but the user may also enter "chi2" to use a
        chi-squared test. Fisher's exact test is more appropriate when the expected
        number of edges are small.

    Returns
    -------
    DensityTestResult: namedtuple
        This named tuple returns the following data:
        stat: float
            The statistic for the test specified by ``method``.
        pvalue: float
            The p-value for the test specified by ``method``.
        misc: dict
            Dictionary containing a number of computed statistics for the network
            comparison performed:
            "probability1" = float
                The probability of an edge (density) in network 1 (p1).
            "probability2" = float
                The probability of an edge (density) in network 2 (p2).
            "observed1" = n_observed1, dataframe
                The total number of edge connections for network 1.
            "observed2" = n_observed2, dataframe
                The total number of edge connections for network 2.
            "possible1" = n_possible1, dataframe
                The total number of possible edges for network 1.
            "possible2" = n_possible2, dataframe
                The total number of possible edges for network 1.

    Notes
    -----
    This test makes several assumptions about the data and test (which could easily be
    loosened in future versions):
        - We assume that the networks are directed. If the networks are undirected (and
        the adjacency matrices are thus symmetric), then edges would be counted twice,
        which would lead to an incorrect calculation of the edge probability. We believe
        passing in the upper or lower triangle of the adjacency matrix would solve this,
        but this has not been tested.
        - We assume that the networks are loopless, that is we do not consider the
        probability of an edge existing between a node and itself. This can be weakened
        and made an option in future versions.
        - We only implement the alternative hypothesis of "not equals" (two-sided);
        future versions could implement the one-sided alternative hypotheses.
    """
    stat, pvalue, misc = group_connection_test(
        A1,
        A2,
        labels1=np.ones(A1.shape[0]),
        labels2=np.ones(A2.shape[0]),
        method=method,
    )
    misc["probability1"] = misc["probabilities1"]
    del misc["probabilities1"]
    misc["probability2"] = misc["probabilities2"]
    del misc["probabilities2"]

    return DensityTestResult(stat, pvalue, misc)
