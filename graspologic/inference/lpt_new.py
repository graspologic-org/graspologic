# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
from scipy.linalg import orthogonal_procrustes
from ..embed import AdjacencySpectralEmbed, OmnibusEmbed, select_dimension
from ..simulations import rdpg
from ..utils import import_graph, is_symmetric


def lpt_function(
    A1, A2, embedding="ase", n_components=None, n_bootstraps=500, test_case="rotation"
):
    if type(embedding) is not str:
        raise TypeError("embedding must be str")
    if type(n_bootstraps) is not int:
        raise TypeError()
    if type(test_case) is not str:
        raise TypeError()
    if n_bootstraps < 1:
        raise ValueError(
            "{} is invalid number of bootstraps, must be greater than 1".format(
                n_bootstraps
            )
        )
    if embedding not in ["ase", "omnibus"]:
        raise ValueError("{} is not a valid embedding method.".format(embedding))
    if test_case not in ["rotation", "scalar-rotation", "diagonal-rotation"]:
        raise ValueError(
            "test_case must be one of 'rotation', 'scalar-rotation',"
            + "'diagonal-rotation'"
        )

    A1 = import_graph(A1)
    A2 = import_graph(A2)
    if not is_symmetric(A1) or not is_symmetric(A2):
        raise NotImplementedError()  # TODO asymmetric case
    if A1.shape != A2.shape:
        raise ValueError("Input matrices do not have matching dimensions")
    if n_components is None:
        # get the last elbow from ZG for each and take the maximum
        num_dims1 = select_dimension(A1)[0][-1]
        num_dims2 = select_dimension(A2)[0][-1]
        n_components = max(num_dims1, num_dims2)
    X_hats = embed(A1, A2, embedding, n_components)
    sample_T_statistic = difference_norm(X_hats[0], X_hats[1], embedding, test_case)
    null_distribution_1 = bootstrap(
        X_hats[0], embedding, n_components, n_bootstraps, test_case
    )
    null_distribution_2 = bootstrap(
        X_hats[1], embedding, n_components, n_bootstraps, test_case
    )

    # using exact mc p-values (see, for example, Phipson and Smyth, 2010)
    p_value_1 = (
        len(null_distribution_1[null_distribution_1 >= sample_T_statistic]) + 1
    ) / (n_bootstraps + 1)
    p_value_2 = (
        len(null_distribution_2[null_distribution_2 >= sample_T_statistic]) + 1
    ) / (n_bootstraps + 1)

    p_value = max(p_value_1, p_value_2)

    misc_stats = {
        "null_distribution_1": null_distribution_1,
        "null_distribution_2": null_distribution_2,
        "p_value_1": p_value_1,
        "p_value_2": p_value_2,
    }

    return p_value, sample_T_statistic, misc_stats


def bootstrap(
    X_hat, embedding, n_components, n_bootstraps, test_case, rescale=False, loops=False
):
    t_bootstrap = np.zeros(n_bootstraps)
    for i in range(n_bootstraps):
        A1_simulated = rdpg(X_hat, rescale=rescale, loops=loops)
        A2_simulated = rdpg(X_hat, rescale=rescale, loops=loops)
        X1_hat_simulated, X2_hat_simulated = embed(
            A1_simulated, A2_simulated, embedding, n_components, check_lcc=False
        )
        t_bootstrap[i] = difference_norm(
            X1_hat_simulated, X2_hat_simulated, embedding, test_case
        )
    return t_bootstrap


def difference_norm(X1, X2, embedding, test_case):
    if embedding in ["ase"]:
        if test_case == "rotation":
            R = orthogonal_procrustes(X1, X2)[0]
            return np.linalg.norm(X1 @ R - X2)
        elif test_case == "scalar-rotation":
            R, s = orthogonal_procrustes(X1, X2)
            return np.linalg.norm(s / np.sum(X1 ** 2) * X1 @ R - X2)
        elif test_case == "diagonal-rotation":
            normX1 = np.sum(X1 ** 2, axis=1)
            normX2 = np.sum(X2 ** 2, axis=1)
            normX1[normX1 <= 1e-15] = 1
            normX2[normX2 <= 1e-15] = 1
            X1 = X1 / np.sqrt(normX1[:, None])
            X2 = X2 / np.sqrt(normX2[:, None])
            R = orthogonal_procrustes(X1, X2)[0]
            return np.linalg.norm(X1 @ R - X2)
    else:
        # in the omni case we don't need to align
        return np.linalg.norm(X1 - X2)


def embed(A1, A2, embedding, n_components, check_lcc=True):
    if embedding == "ase":
        X1_hat = AdjacencySpectralEmbed(
            n_components=n_components, check_lcc=check_lcc
        ).fit_transform(A1)
        X2_hat = AdjacencySpectralEmbed(
            n_components=n_components, check_lcc=check_lcc
        ).fit_transform(A2)
    elif embedding == "omnibus":
        X_hat_compound = OmnibusEmbed(
            n_components=n_components, check_lcc=check_lcc
        ).fit_transform((A1, A2))
        X1_hat = X_hat_compound[0]
        X2_hat = X_hat_compound[1]
    return (X1_hat, X2_hat)

