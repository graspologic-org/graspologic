import numpy as np


def aLAP(cost_matrix, maximize=True):
    """
    An approximation algorithm for solving the Linear Assignment problem

    Parameters
    ----------
    cost_matrix : array
        The cost matrix of the bipartite graph.

    maximize : bool (default: True)
        Calculates a minimum weight matching if false.

    Returns
    -------
    row_ind, col_ind : array
        An array of row indices and one of corresponding column indices giving
        the optimal assignment. The cost of the assignment can be computed
        as ``cost_matrix[row_ind, col_ind].sum()``. The row indices will be
        sorted; in the case of a square cost matrix they will be equal to
        ``numpy.arange(cost_matrix.shape[0])``.
    """

    if not maximize:
        cost_matrix = -cost_matrix
    num_vert = cost_matrix.shape[0]
    n = 2 * num_vert
    matched = np.empty(n) * np.nan
    cv = np.zeros(n)
    qn = np.zeros(n)
    col_argmax = np.argmax(cost_matrix, axis=0)
    row_argmax = np.argmax(cost_matrix, axis=1)

    # remove full zero rows and columns (match them)
    col_z = np.count_nonzero(cost_matrix, axis=0)
    col_z = np.arange(num_vert)[col_z == np.zeros(num_vert)]
    row_z = np.count_nonzero(cost_matrix, axis=1)
    row_z = np.arange(num_vert)[row_z == np.zeros(num_vert)]
    mz = min([len(row_z), len(col_z)])
    col_z = col_z[:mz]
    row_z = row_z[:mz]

    cv[:num_vert] = col_argmax + num_vert
    # first half points to second, vice versa
    cv[num_vert:] = row_argmax
    cv[col_z] = row_z + num_vert
    cv[row_z + num_vert] = col_z
    cv = cv.astype(int)

    dom_ind = cv[cv] == np.arange(n)
    matched[dom_ind] = cv[dom_ind]  # matched indices, everywhere else nan
    qc, = np.nonzero(dom_ind)  # dominating vertices

    while (
        len(qc) > 0 and np.isnan(matched).any()
    ):  # loop while qc not empty, ie new matchings still being found

        temp = np.arange(n)[np.in1d(cv, qc)]  # indices of qc in cv
        qt = temp[
            ~np.in1d(temp, matched[qc])
        ]  # indices of unmatched verts in cv and qc

        qt_p = qt[qt >= num_vert]
        qt_n = qt[qt < num_vert]

        m_row = np.arange(num_vert)[
            np.isnan(matched[num_vert:])
        ]  # unmatched rows to check
        m_col = np.arange(num_vert)[np.isnan(matched[:num_vert])]
        # unmatched cols

        col_argmax = np.argmax(cost_matrix[np.ix_(m_row, qt_n)], axis=0)
        row_argmax = np.argmax(cost_matrix[np.ix_(qt_p - num_vert, m_col)], axis=1)

        col_argmax = m_row[col_argmax]
        row_argmax = m_col[row_argmax]

        cv[qt_n] = col_argmax + num_vert
        cv[qt_p] = row_argmax
        cv = cv.astype(int)

        dom_ind = cv[cv[qt]] == qt
        qt = qt[dom_ind]
        matched[qt] = cv[qt]  # adding new dominating indices to matching
        matched[cv[qt]] = qt

        qn = np.zeros(n)  # store new matchings
        qn[qt] = qt
        qn[cv[qt]] = cv[qt]
        qc = qn[qn > 0].astype(int)

    matching = matched[num_vert:]
    rows = np.arange(num_vert)[~np.isnan(matching)]
    matching = matching[~np.isnan(matching)].astype(int)

    return (rows, matching)
