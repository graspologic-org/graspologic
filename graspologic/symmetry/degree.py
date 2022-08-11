from scipy.stats import combine_pvalues, ks_2samp


def degree_test(A1, A2, method="ks"):
    in_degrees1 = A1.sum(axis=0)
    out_degrees1 = A1.sum(axis=1)

    in_degrees2 = A2.sum(axis=0)
    out_degrees2 = A2.sum(axis=1)

    # # this is dumb in the sense that p-values aren't independent
    # in_stat, in_pvalue = ks_2samp(in_degrees1, in_degrees2, alternative="two-sided")
    # out_stat, out_pvalue = ks_2samp(out_degrees1, out_degrees2, alternative="two-sided")
    # stat, pvalue = combine_pvalues((in_pvalue, out_pvalue))

    stat, pvalue = ks_2samp(
        in_degrees1 + out_degrees2, in_degrees2 + out_degrees2, alternative="two-sided"
    )

    return stat, pvalue, {}
