import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.contingency_tables import mcnemar

from ..types import AdjacencyMatrix, GraphRepresentation
from .fisher_exact_nonunity import fisher_exact_nonunity


def binom_2samp(
    x1: int, n1: int, x2: int, n2: int, null_ratio: float = 1.0, method: str = "fisher"
) -> tuple[float, float]:
    """
    This function computes the likelihood that two binomial samples are drown from identical underlying
    distributions. Null hypothesis is that the success probability for each sample is identical (i.e.
    p1 = p2), and this function returns the probability that the null hypothesis is accurate, under a
    variety of potential statistical tests (default is Fisher's exact test). This test can be used to
    compare graph density by counting the total nunmber of actual edges and total number of possible
    edges in two different graphs. Given these parameters, this function will compute the probability of
    the observed actual edge counts for each of the two different graphs assuming that the null
    hypothesis (i.e. the densities of both graphs are equal) is true.

    Parameters
    ----------
    x1 : int
        Success count for group 1, i.e. the number of actual edges between nodes in graph 1
    n1 : int
        The number of possible edges between nodes in graph 1 (i.e. the edge count if the graph were complete)
    x2 : int
        Success count for group 2, i.e. the number of actual edges between nodes in graph 2
    n2 : total possible in group 2
        The number of possible edges between nodes in graph 2 (i.e. the edge count if the graph were complete)
    null_ratio : float, optional
        Optional parameter for testing whether p1 is a fixed ratio larger or smaller than p2, i.e. p1 = cp2,
        where c is the null_ratio. Defatault is 1.0. This parameter can only be !=1 if the chosen statistical
        test is Fisher's exact test.
    method : str, optional
        Defines the statistical test to be run in order to reject or fail to reject the null hypothesis.
        By default, this is the Fisher's exact test (i.e. "fisher"). The chi-squared test is also an option if
        the user enters "chi2".

    Returns
    -------
    stat: float
        The odds ratio for the provided data, representing the prior probability of a "success" (in this
        case, the odds of an edge occurring between two nodes)
    pvalue: float
        The computed probability of the observed dataset assuming the null hypothesis (p1=p2) is true. By
        convention, a pvalue < 0.05 represents a statistically significant result.

    Raises
    ------
    ValueError: "Non-unity null odds only works with Fisher's exact test"
        Only the Fisher's exact test allows us to test whether p1 is a fixed multiple of p2, i.e. if the
        null_ratio != 1. If this function is called where a null ratio other than 1 is entered and a test
        other than "fisher" is chosen, the function will throw the above error.
    ValueError
        This function will also throw a ValueError if the value entered for the method, i.e. the test
        to be performed, is anything other than "fisher" or "chi2."

    References
    ------
    [1] Alan Agresti. Categorical data analysis. John Wiley & Sons, 3 edition, 2013.

    Notes
    ------
    The version of this file in github/neurodata/bilateral-connectome included a number of other possible
    statistical tests. This version discards all except Fisher and chi-squared, because the others proved
    less useful when testing this code.
    """
    if x1 == 0 or x2 == 0:
        # logging.warn("One or more counts were 0, not running test and returning nan")
        return np.nan, np.nan
    if null_ratio != 1 and method != "fisher":
        raise ValueError("Non-unity null odds only works with Fisher's exact test")

    cont_table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    if method == "fisher" and null_ratio == 1.0:
        stat, pvalue = fisher_exact(cont_table, alternative="two-sided")
    elif method == "fisher" and null_ratio != 1.0:
        stat, pvalue = fisher_exact_nonunity(cont_table, null_ratio=null_ratio)
    elif method == "chi2":
        stat, pvalue, _, _ = chi2_contingency(cont_table)
    else:
        raise ValueError()

    return stat, pvalue


def binom_2samp_paired(
    x1: GraphRepresentation, x2: GraphRepresentation
) -> tuple[float, float, dict]:
    """
    Similar to the above, except this function applies in the case where the two binomial samples are
    paired. In the case of graph comparisons, this could involve, e.g., an identical underlying set of
    nodes but two different sets of edges. Again, the goal is to assess the probability of the observed
    samples assuming the underlying binomial distribution is equal for both measured samples.

    Parameters
    ----------
    x1 : array, shape varies
        An array of either zeroes or ones representing either succcesses (edge present) or failures (edge
        absent) for the first sample for all possible edges between nodes in the underlying graph. The
        size will correspond to the size of the graph or subgraph upon which the test is being performed
        (this is supplied by the user).
    x2 : array, shape varies but must equal shape of x1
        Same as x1, but for the second sample. The shape of this array must match the shape of the array
        for sample 1 in order for paired testing to be possible.

    Returns
    -------
    stat: float
        A statistic computed as part of McNemar's test. Returns the smaller of n_only_x1 and n_only_x2,
        where these values represent the number of 1s present in one array and absent from the other.
    pvalue: float
        The computed probability of the observed dataset assuming the null hypothesis (p1=p2) is true. By
        convention, a pvalue < 0.05 represents a statistically significant result.
    misc:  dict
        Dictionary containing a few miscellaneous statistics computed by the function:
            "n_both" contains the number of locations where a 1 appears in both x1 and x2
            "n_neither" contains the number of locations where a 0 appears in both x1 and x2
            "n_only1" contains the number of locations where a 1 appears in x1 but not x2
            "n_only2" contains the number of locations where a 1 appears in x2 but not x1

    """

    # x1 = x1.astype(bool)
    # x2 = x2.astype(bool)

    # TODO these two don't actually matter at all for McNemar's test...
    # n_both = (x1 & x2).sum()
    n_both = np.sum(x1 + x2 == 2)
    # n_neither = ((~x1) & (~x2)).sum()
    n_neither = np.sum(x1 + x2 == 0)

    # n_only_x1 = (x1 & (~x2)).sum()
    n_only_x1 = np.sum(x1 - x2 == 1)
    # n_only_x2 = ((~x1) & x2).sum()
    n_only_x2 = np.sum(x2 - x1 == 1)

    cont_table = np.array([[n_both, n_only_x2], [n_only_x1, n_neither]])
    # cont_table = np.array(cont_table)

    bunch = mcnemar(cont_table)
    stat = bunch.statistic
    pvalue = bunch.pvalue

    misc = {}
    misc["n_both"] = n_both
    misc["n_neither"] = n_neither
    misc["n_only1"] = n_only_x1
    misc["n_only2"] = n_only_x2

    return stat, pvalue, misc
