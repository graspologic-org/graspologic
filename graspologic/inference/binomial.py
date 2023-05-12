from collections import namedtuple
from typing import Literal

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import test_proportions_2indep

BinomialResult = namedtuple("BinomialResult", ["stat", "pvalue"])
BinomialTestMethod = Literal["fisher", "chi2", 'score']


def binom_2samp(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    null_ratio: float = 1.0,
    method: BinomialTestMethod = "fisher",
) -> BinomialResult:
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
        where c is the null_ratio. Default is 1.0. This parameter can only be !=1 if the chosen statistical
        test is Fisher's exact test.
    method : str, optional
        Defines the statistical test to be run in order to reject or fail to reject the null hypothesis.
        By default, this is the Fisher's exact test (i.e. "fisher"). The chi-squared test is also an option if
        the user enters "chi2".

    Returns
    -------
    BinomialResult: namedtuple
    This namedtuple contains the following data:
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
        return BinomialResult(np.nan, np.nan)
    if null_ratio != 1 and method != "score":
        raise ValueError("Non-unity null odds only works with ``method=='score'``")

    cont_table = np.array([[x1, n1 - x1], [x2, n2 - x2]])
    if method == "fisher" and null_ratio == 1.0:
        stat, pvalue = fisher_exact(cont_table, alternative="two-sided")
    elif method == "chi2":
        stat, pvalue, _, _ = chi2_contingency(cont_table)
    elif method == 'score': 
        stat, pvalue = test_proportions_2indep(
            x1,
            n1,
            x2,
            n2,
            method="score",
            compare="ratio",
            alternative="two-sided",
            value=null_ratio,
        )
    else:
        raise ValueError()

    return BinomialResult(stat, pvalue)
