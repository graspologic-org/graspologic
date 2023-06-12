from collections import namedtuple

import numpy as np
from beartype import beartype

from ..types import AdjacencyMatrix
from .binomial import BinomialTestMethod
from .group_connection_test import group_connection_test

DensityTestResult = namedtuple("DensityTestResult", ["stat", "pvalue", "misc"])


@beartype
def density_test(
    A1: AdjacencyMatrix, A2: AdjacencyMatrix, method: BinomialTestMethod = "fisher"
) -> DensityTestResult:
    r"""
    Compares two networks by testing whether the global connection probabilities
    (densites) for the two networks are equal under an Erdos-Renyi model assumption.

    Parameters
    ----------
    A1: np.array, int
        The adjacency matrix for network 1. Will be treated as a binary network,
        regardless of whether it was weighted.
    A2: np.array, int
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

                "probability1", float
                    The probability of an edge (density) in network 1 (:math:`p_1`).
                "probability2", float
                    The probability of an edge (density) in network 2 (:math:`p_2`).
                "observed1", pd.DataFrame
                    The total number of edge connections for network 1.
                "observed2", pd.DataFrame
                    The total number of edge connections for network 2.
                "possible1", pd.DataFrame
                    The total number of possible edges for network 1.
                "possible2", pd.DataFrame
                    The total number of possible edges for network 1.


    Notes
    -----
    Under the Erdos-Renyi model, edges are generated independently with probability
    :math:`p`. :math:`p` is also known as the network density. This function tests
    whether the probability of an edge in network 1 (:math:`p_1`) is significantly
    different from that in network 2 (:math:`p_2`), by assuming that both networks came
    from an Erdos-Renyi model. In other words, the null hypothesis is

    .. math:: H_0: p_1 = p_2

    And the alternative hypothesis is

    .. math:: H_A: p_1 \neq p_2

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
