import numpy as np
from scipy.stats import beta, chi2
from scipy.stats import combine_pvalues as scipy_combine_pvalues
from scipy.stats import ks_1samp, uniform
from beartype import beartype

@beartype
def combine_pvalues(pvalues, method="fisher"):
    pvalues: list
    method: str

    """
    Takes the computed p-values from the stochastic block model test (i.e. the per-group p values) and combines them into a single p-value
    for the hypothesis that one group-to-group adjacency matrix is statistically the same as a second group-to-group adjacency matrix. 
    There are multiple methods for performing this calculation, and this function permits a number of methods to be selected (default
    is Fisher's test.

    Parameters
    ----------
    pvalues:    float, list size((number of groups)^2)
        These are the computed p-values for all group-to-group connection comparisons performed by the stochastic block model. The size of
        the list corresponds to the square of the total number of groups to which each node can belong.

    method:     string
        The statistical method to be used to combine the p-values. Default is Fisher's test, but the user may also select "tippett" to use
        Tippett's method. Any entry other than "fisher" or "tippett" will throw an error.

    Returns
    -------
    stat: float
        A statistic computed as part of the statistical test performed. For Tippett's method, this is simply the least of the p-values. For 
        Fisher's method, this is the test statistic computed as -2*sum(log(p-values)). 
    
    pvalue
        The combined p-value computed according to the chosen method. 

    Raises
    ------
    NotImplementedError()
        Occurs if the user chooses a method other than fisher or tippett.

    Note
    ------
    The bilateral-connectome implementation includes numerous options for "method." This version discards all except fisher and tippett,
    as these proved most useful when actual operating this code.

    scipy_combine_pvalues has a bug for Tippett's method in any release prior to version 1.9. The user must download release 1.9 in order
    for the code below to function properly.
    """
    pvalues = np.array(pvalues)

    # scipy formerly had a bug for Tippett's method but this is fixed as of release 1.9
    if method == "tippett":  
        stat, pvalue = scipy_combine_pvalues(pvalues, method="tippett")
    elif method == "fisher":
        stat, pvalue = scipy_combine_pvalues(pvalues, method="fisher")
    else:
        raise NotImplementedError()

    return stat, pvalue
