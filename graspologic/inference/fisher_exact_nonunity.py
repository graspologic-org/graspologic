from collections import namedtuple

import numpy as np
from scipy.stats import nchypergeom_fisher

from ..types import GraphRepresentation

FisherResult = namedtuple("FisherResult", ["oddsratio", "pvalue"])


def fisher_exact_nonunity(
    table: GraphRepresentation,
    alternative: str = "two-sided",
    null_odds_ratio: float = 1.0,
) -> FisherResult:
    """
    Perform a Fisher exact test on a 2x2 contingency table. When testing whether two networks are statistically distinct, the rows of the
    table correspond to the two networks, and the columns correspond to the number of actual edges and the number of places where an edge
    could be but no edge is found. In other words, row 1 corresponds to network 1, the first entry in the row corresponds to the number of
    edges in network 1, and the second entry corresponds to the (total number of possible edges - number of actual edges). The second row
    has the same information for the second network.

    Parameters
    ----------
    table : array_like of ints
        A 2x2 contingency table, meeting the criteria discussed above.
    alternative : {'two-sided', 'less', 'greater'}, optional
        This parameter defines the alternative hypothesis for the statistical test. "two-sided", the default, tests whether the second
        network's density is greater or less than the first network's density. If "less" or "greater" is selected, then the function
        performs a one-sided test to determine whether the second network's density is strictly less than- or strictly greater than the
        first network's.
    null_odds_ratio : float, optional (default=1)
        A (possibly non-unity) null odds ratio. This parameter can be set to a value other than 1 to test a null hypothesis that
        the odds ratio for the second network is a fixed multiple of the first network's odd ratio.

    Returns
    -------
    FisherResult: namedtuple
        A namedtuple containing the following data:
    oddsratio : float
        The odds for network 1 are calculated as (edge present)/(edge absent). The odds for network 2 are computed the same way, and the
        odds ratio variable returns (odds for network 1)/(odds for network 2).
    p_value : float
        The probability of obtaining a contingency table at least as extreme as the one actually observed, assuming the null hypothesis is
        correct.


    Notes
    -----
    For further information regarding fisher's exact test, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.fisher_exact.html.

    """
    dist = nchypergeom_fisher

    # int32 is not enough for the algorithm
    c = np.asarray(table, dtype=np.int64)
    if not c.shape == (2, 2):
        raise ValueError("The input `table` must be of shape (2, 2).")

    if np.any(c < 0):
        raise ValueError("All values in `table` must be nonnegative.")

    if 0 in c.sum(axis=0) or 0 in c.sum(axis=1):
        # If both values in a row or column are zero, the p-value is 1 and
        # the odds ratio is NaN.
        return FisherResult(np.nan, 1.0)

    if c[1, 0] > 0 and c[0, 1] > 0:
        oddsratio = c[0, 0] * c[1, 1] / (c[1, 0] * c[0, 1])
    else:
        oddsratio = np.inf

    n1 = c[0, 0] + c[0, 1]
    n2 = c[1, 0] + c[1, 1]
    n = c[0, 0] + c[1, 0]

    rv = dist(n1 + n2, n1, n, null_odds_ratio)

    def binary_search(n: int, n1: int, n2: int, side: str) -> int:
        """Binary search for where to begin halves in two-sided test."""
        if side == "upper":
            minval = mode
            maxval = n
        else:
            minval = 0
            maxval = mode
        guess = -1
        while maxval - minval > 1:
            if maxval == minval + 1 and guess == minval:
                guess = maxval
            else:
                guess = (maxval + minval) // 2
            pguess = rv.pmf(guess)
            if side == "upper":
                ng = guess - 1
            else:
                ng = guess + 1
            if pguess <= pexact < rv.pmf(ng):
                break
            elif pguess < pexact:
                maxval = guess
            else:
                minval = guess
        if guess == -1:
            guess = minval
        if side == "upper":
            while guess > 0 and rv.pmf(guess) < pexact * epsilon:
                guess -= 1
            while rv.pmf(guess) > pexact / epsilon:
                guess += 1
        else:
            while rv.pmf(guess) < pexact * epsilon:
                guess += 1
            while guess > 0 and rv.pmf(guess) > pexact / epsilon:
                guess -= 1
        return guess

    if alternative == "less":
        pvalue = rv.cdf(c[0, 0])
    elif alternative == "greater":
        # Same formula as the 'less' case, but with the second column.
        pvalue = rv.sf(c[0, 0] - 1)
    elif alternative == "two-sided":
        mode = int((n + 1) * (n1 + 1) / (n1 + n2 + 2))
        pexact = dist.pmf(c[0, 0], n1 + n2, n1, n, null_odds_ratio)
        pmode = dist.pmf(mode, n1 + n2, n1, n, null_odds_ratio)

        epsilon = 1 - 1e-4
        if np.abs(pexact - pmode) / np.maximum(pexact, pmode) <= 1 - epsilon:
            return FisherResult(oddsratio, 1.0)

        elif c[0, 0] < mode:
            plower = dist.cdf(c[0, 0], n1 + n2, n1, n, null_odds_ratio)
            if dist.pmf(n, n1 + n2, n1, n, null_odds_ratio) > pexact / epsilon:
                return FisherResult(oddsratio, plower)

            guess = binary_search(n, n1, n2, "upper")
            pvalue = plower + dist.sf(guess - 1, n1 + n2, n1, n, null_odds_ratio)
        else:
            pupper = dist.sf(c[0, 0] - 1, n1 + n2, n1, n, null_odds_ratio)
            if dist.pmf(0, n1 + n2, n1, n, null_odds_ratio) > pexact / epsilon:
                return FisherResult(oddsratio, pupper)

            guess = binary_search(n, n1, n2, "lower")
            pvalue = pupper + dist.cdf(guess, n1 + n2, n1, n, null_odds_ratio)
    else:
        msg = "`alternative` should be one of {'two-sided', 'less', 'greater'}"
        raise ValueError(msg)

    pvalue = min(pvalue, 1.0)

    return FisherResult(oddsratio, pvalue)
