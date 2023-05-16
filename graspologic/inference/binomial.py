from collections import namedtuple
from typing import Literal

import numpy as np
from scipy.stats import chi2_contingency, fisher_exact
from statsmodels.stats.proportion import test_proportions_2indep

BinomialResult = namedtuple("BinomialResult", ["stat", "pvalue"])
BinomialTestMethod = Literal["score", "fisher", "chi2"]


def binom_2samp(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    null_ratio: float = 1.0,
    method: BinomialTestMethod = "score",
) -> BinomialResult:
    """
    This function computes the likelihood that two binomial samples are drown from
    identical underlying distributions. Null hypothesis is that the success probability
    for each sample is identical (i.e. p1 = p2), and this function returns the
    probability that the null hypothesis is accurate, under a variety of potential
    statistical tests (default is score test).

    Parameters
    ----------
    x1 : int
        Success count for group 1
    n1 : int
        The number of possible successes for group 1
    x2 : int
        Success count for group 2
    n2 : int
        The number of possible successes for group 2
    null_ratio : float, optional
        Optional parameter for testing whether p1 is a fixed ratio larger or smaller
        than p2, i.e. p1 = cp2, where c is the null_ratio. Default is 1.0. This
        parameter can only be !=1 if the chosen statistical test is the score test.
    method : str, optional
        Defines the statistical test to be run in order to reject or fail to reject the
        null hypothesis. By default, this is the score test (i.e. "score").

    Returns
    -------
    BinomialResult: namedtuple
    This namedtuple contains the following data:
        stat: float
            Test statistic for the requested test.
        pvalue: float
            The p-value for the requested test.

    References
    ------
    [1] Alan Agresti. Categorical data analysis. John Wiley & Sons, 3 edition, 2013.

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
    elif method == "score":
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
