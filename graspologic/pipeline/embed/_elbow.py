# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.


from typing import List, Tuple, Union

import numpy as np
from scipy.stats import norm


def _compute_likelihood(arr):
    """
    Computes the log likelihoods based on normal distribution given
    a 1d-array of sorted values. If the input has no variance,
    the likelihood will be nan.
    """
    n_elements = len(arr)
    likelihoods = np.zeros(n_elements)

    for idx in range(1, n_elements + 1):
        # split into two samples
        s1 = arr[:idx]
        s2 = arr[idx:]

        # deal with when input only has 2 elements
        if (s1.size == 1) & (s2.size == 1):
            likelihoods[idx - 1] = -np.inf
            continue

        # compute means
        mu1 = np.mean(s1)
        if s2.size != 0:
            mu2 = np.mean(s2)
        else:
            # Prevent numpy warning for taking mean of empty array
            mu2 = -np.inf

        # compute pooled variance
        variance = ((np.sum((s1 - mu1) ** 2) + np.sum((s2 - mu2) ** 2))) / (
            n_elements - 1 - (idx < n_elements)
        )
        std = np.sqrt(variance)

        # compute log likelihoods
        likelihoods[idx - 1] = np.sum(norm.logpdf(s1, loc=mu1, scale=std)) + np.sum(
            norm.logpdf(s2, loc=mu2, scale=std)
        )

    return likelihoods


def _find_elbows(priority_ordered_matrix: np.ndarray, n_elbows: int) -> List[int]:
    # use Ghodsi & Zhu method for finding elbow
    idx = 0
    elbows = []
    for _ in range(n_elbows):
        arr = priority_ordered_matrix[idx:]
        if arr.size <= 1:  # Cant compute likelihoods with 1 numbers
            break
        lq = _compute_likelihood(arr)
        idx += np.argmax(lq) + 1
        elbows.append(idx)

    return elbows


def _index_of_elbow(
    priority_ordered_matrix: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
    n_elbows: int,
) -> int:
    if isinstance(priority_ordered_matrix, tuple):
        left_elbows = _find_elbows(priority_ordered_matrix[0], n_elbows)
        right_elbows = _find_elbows(priority_ordered_matrix[1], n_elbows)
        return max(left_elbows[-1], right_elbows[-1])
    else:
        elbows = _find_elbows(priority_ordered_matrix, n_elbows)
        return elbows[-1]
