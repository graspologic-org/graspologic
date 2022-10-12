from collections import namedtuple

import numpy as np

from ..types import GraphRepresentation
from .group_connection_test import group_connection_test

DensityTestResult = namedtuple("DensityTestResult", ["stat", "pvalue", "er_misc"])


def density_test(
    A1: GraphRepresentation, A2: GraphRepresentation, method: str = "fisher"
) -> DensityTestResult:

    """
    This function uses the Erdos-Renyi model to perform a density test to compare the adjacency matrices for two networks.
    Under the Erdos-Renyi model, it is assumed that the probability of an edge existing between any two adjacent nodes is equal to some
    constant p. This function tests whether the probability of an edge in network 1 is statistically different from that in network 2.
    In other words, it tests a null hypothesis that p1, the edge probability for network 1, is equal to p2, the edge probability for
    network 2.

    The Erdos-Renyi model is essentially a special case of stochastic block model where all nodes belong to the same group. Thus, to
    perform the required calculations, this function calls group_connection_test, but assigns all nodes to a single group. For further
    information regarding the group_connection_test function, consult the documentation for the .sbm file in this package.

    Parameters
    ----------
    A1: np. array, int
        The adjacency matrix for network 1. Contains either a 0 or 1 at each location in the array, where a 1 denotes an edge and a 0 denotes
        the absence of an edge.
    A2: np. array, int
        Adjacency matrix for network 2.
    method: string, optional, default="fisher"
        Specifies the statistical test to be performed to reject or fail to reject the null hypothesis. The default option is "fisher",
        which uses Fisher's exact test, but the user may also enter "chi2" to use a chi-squared test. Any other entry will give an error.

    Returns
    -------
    DensityTestResult: namedtuple
        This named tuple returns the following data:
        stat: float
          This returns a statistic calculated by group_connection_test when combining p-values for multiple group-to-group comparisons. This
          won't be too meaningful or useful for the Erdos-Renyi test.
       pvalue: float
            The computed probability of the observed network distributions assuming the null hypothesis (i.e. p1 = p2) is correct.
         er_misc: dict
            Dictionary containing a number of computed statistics for the network comparison performed:
            "probability1" = float
                This contains the computed probability of an edge between nodes in network 1. In other words, this is p1
            "probability2" = float
                This contains p2, i.e. the computed network density of network 2.
            "observed1" = n_observed1, dataframe
                The total number of edge connections for network 1.
            "observed2" = n_observed2, dataframe
                Same as above, but for network 2.
            "possible1" = n_possible1, dataframe
                The total number of possible edges for network 1.
            "possible2" = n_possible2, dataframe
                Same as above, but for network 2.

    """
    stat, pvalue, er_misc = group_connection_test(
        A1,
        A2,
        labels1=np.ones(A1.shape[0]),
        labels2=np.ones(A2.shape[0]),
        method=method,
    )
    er_misc["probability1"] = er_misc["probabilities1"]
    del er_misc["probabilities1"]
    er_misc["probability2"] = er_misc["probabilities2"]
    del er_misc["probabilities2"]

    return DensityTestResult(stat, pvalue, er_misc)
