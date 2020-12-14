# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import ParameterGrid

from .base import BaseCluster


class GaussianCluster(BaseCluster):
    r"""
    Gaussian Mixture Model (GMM)

    Representation of a Gaussian mixture model probability distribution.
    This class allows to estimate the parameters of a Gaussian mixture
    distribution. It computes all possible models from one component to
    ``max_components``. The best model is given by the lowest BIC score.

    Parameters
    ----------
    min_components : int, default=2.
        The minimum number of mixture components to consider (unless
        ``max_components`` is None, in which case this is the maximum number of
        components to consider). If ``max_componens`` is not None, ``min_components``
        must be less than or equal to ``max_components``.

    max_components : int or None, default=None.
        The maximum number of mixture components to consider. Must be greater
        than or equal to ``min_components``.

    covariance_type : {'all' (default), 'full', 'tied', 'diag', 'spherical'}, optional
        String or list/array describing the type of covariance parameters to use.
        If a string, it must be one of:

        - 'all'
            considers all covariance structures in ['spherical', 'diag', 'tied', 'full']
        - 'full'
            each component has its own general covariance matrix
        - 'tied'
            all components share the same general covariance matrix
        - 'diag'
            each component has its own diagonal covariance matrix
        - 'spherical'
            each component has its own single variance

        If a list/array, it must be a list/array of strings containing only
            'spherical', 'tied', 'diag', and/or 'full'.

    tol : float, defaults to 1e-3.
        The convergence threshold. EM iterations will stop when the
        lower bound average gain is below this threshold.

    reg_covar : float, defaults to 1e-6.
        Non-negative regularization added to the diagonal of covariance.
        Allows to assure that the covariance matrices are all positive.

    max_iter : int, defaults to 100.
        The number of EM iterations to perform.

    n_init : int, defaults to 1.
        The number of initializations to perform. The best results are kept.

    init_params : {'kmeans', 'random'}, defaults to 'kmeans'.
        The method used to initialize the weights, the means and the
        precisions.
        Must be one of::

            'kmeans' : responsibilities are initialized using kmeans.
            'random' : responsibilities are initialized randomly.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number generator;
        If RandomState instance, ``random_state`` is the random number generator;
        If None, the random number generator is the RandomState instance used
        by ``np.random``.


    Attributes
    ----------
    n_components_ : int
        Optimal number of components based on BIC.
    covariance_type_ : str
        Optimal covariance type based on BIC.
    model_ : GaussianMixture object
        Fitted GaussianMixture object fitted with optimal number of components
        and optimal covariance structure.
    bic_ : pandas.DataFrame
        A pandas DataFrame of BIC values computed for all possible number of clusters
        given by ``range(min_components, max_components + 1)`` and all covariance
        structures given by :attr:`covariance_type`.
    ari_ : pandas.DataFrame
        Only computed when y is given. Pandas Dataframe containing ARI values computed
        for all possible number of clusters given by ``range(min_components,
        max_components)`` and all covariance structures given by :attr:`covariance_type`.
    """

    def __init__(
        self,
        min_components=2,
        max_components=None,
        covariance_type="all",
        tol=1e-3,
        reg_covar=1e-6,
        max_iter=100,
        n_init=1,
        init_params="kmeans",
        random_state=None,
    ):
        if isinstance(min_components, int):
            if min_components <= 0:
                msg = "min_components must be >= 1."
                raise ValueError(msg)
        else:
            msg = "min_components must be an integer, not {}.".format(
                type(min_components)
            )
            raise TypeError(msg)

        if isinstance(max_components, int):
            if max_components <= 0:
                msg = "max_components must be >= 1 or None."
                raise ValueError(msg)
            elif min_components > max_components:
                msg = "min_components must be less than or equal to max_components."
                raise ValueError(msg)
        elif max_components is not None:
            msg = "max_components must be an integer or None, not {}.".format(
                type(max_components)
            )
            raise TypeError(msg)

        if isinstance(covariance_type, (np.ndarray, list)):
            covariance_type = np.unique(covariance_type)
        elif isinstance(covariance_type, str):
            if covariance_type == "all":
                covariance_type = ["spherical", "diag", "tied", "full"]
            else:
                covariance_type = [covariance_type]
        else:
            msg = "covariance_type must be a numpy array, a list, or "
            msg += "string, not {}".format(type(covariance_type))
            raise TypeError(msg)

        for cov in covariance_type:
            if cov not in ["spherical", "diag", "tied", "full"]:
                msg = (
                    "covariance structure must be one of "
                    + '["spherical", "diag", "tied", "full"]'
                )
                msg += " not {}".format(cov)
                raise ValueError(msg)

        new_covariance_type = []
        for cov in ["spherical", "diag", "tied", "full"]:
            if cov in covariance_type:
                new_covariance_type.append(cov)

        self.min_components = min_components
        self.max_components = max_components
        self.covariance_type = new_covariance_type
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fits gaussian mixure model to the data.
        Estimate model parameters with the EM algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.

        y : array-like, shape (n_samples,), optional (default=None)
            List of labels for X if available. Used to compute
            ARI scores.

        Returns
        -------
        self
        """

        # Deal with number of clusters
        if self.max_components is None:
            lower_ncomponents = 1
            upper_ncomponents = self.min_components
        else:
            lower_ncomponents = self.min_components
            upper_ncomponents = self.max_components

        n_mixture_components = upper_ncomponents - lower_ncomponents + 1

        if upper_ncomponents > X.shape[0]:
            if self.max_components is None:
                msg = "if max_components is None then min_components must be >= "
                msg += "n_samples, but min_components = {}, n_samples = {}".format(
                    upper_ncomponents, X.shape[0]
                )
            else:
                msg = "max_components must be >= n_samples, but max_components = "
                msg += "{}, n_samples = {}".format(upper_ncomponents, X.shape[0])
            raise ValueError(msg)
        elif lower_ncomponents > X.shape[0]:
            msg = "min_components must be <= n_samples, but min_components = "
            msg += "{}, n_samples = {}".format(upper_ncomponents, X.shape[0])
            raise ValueError(msg)

        # Get parameters
        random_state = self.random_state

        param_grid = dict(
            covariance_type=self.covariance_type,
            n_components=range(lower_ncomponents, upper_ncomponents + 1),
            tol=[self.tol],
            reg_covar=[self.reg_covar],
            max_iter=[self.max_iter],
            n_init=[self.n_init],
            init_params=[self.init_params],
            random_state=[random_state],
        )

        param_grid = list(ParameterGrid(param_grid))

        models = [[] for _ in range(n_mixture_components)]
        bics = [[] for _ in range(n_mixture_components)]
        aris = [[] for _ in range(n_mixture_components)]

        for i, params in enumerate(param_grid):
            model = GaussianMixture(**params)
            model.fit(X)
            models[i % n_mixture_components].append(model)
            bics[i % n_mixture_components].append(model.bic(X))
            if y is not None:
                predictions = model.predict(X)
                aris[i % n_mixture_components].append(
                    adjusted_rand_score(y, predictions)
                )

        self.bic_ = pd.DataFrame(
            bics,
            index=np.arange(lower_ncomponents, upper_ncomponents + 1),
            columns=self.covariance_type,
        )

        if y is not None:
            self.ari_ = pd.DataFrame(
                aris,
                index=np.arange(lower_ncomponents, upper_ncomponents + 1),
                columns=self.covariance_type,
            )
        else:
            self.ari_ = None

        # Get the best cov type and its index within the dataframe
        best_covariance = self.bic_.min(axis=0).idxmin()
        best_covariance_idx = self.covariance_type.index(best_covariance)

        # Get the index best component for best_covariance
        best_component = self.bic_.idxmin()[best_covariance]

        self.n_components_ = best_component
        self.covariance_type_ = best_covariance
        self.model_ = models[best_component - self.min_components][best_covariance_idx]

        return self
