# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
    max_components. The best model is given by the lowest BIC score.

    Parameters
    ----------
    min_components : int, defaults to 1. 
        The minimum number of mixture components to consider (unless
        max_components=None, in which case this is the maximum number of
        components to consider). If max_componens is not None, min_components
        must be less than or equal to max_components.
    max_components : int, defaults to 1.
        The maximum number of mixture components to consider. Must be greater 
        than or equal to min_components.

    covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}, optional
        String or list/array describing the type of covariance parameters to use.
        If a string, it must be one of:
        
        - 'full'
            each component has its own general covariance matrix
        - 'tied'
            all components share the same general covariance matrix
        - 'diag'
            each component has its own diagonal covariance matrix
        - 'spherical'
            each component has its own single variance
        - 'all'
            considers all covariance structures in ['spherical', 'diag', 'tied', 'full']
        If a list/array, it must be a list/array of strings containing only
            'spherical', 'tied', 'diag', and/or 'spherical'.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by ``np.random``.

    Attributes
    ----------
    n_components_ : int
        Optimal number of components based on BIC.
    model_ : GaussianMixture object
        Fitted GaussianMixture object fitted with optimal numeber of components 
        and optimal covariance structure.
    bic_ : pandas.DataFrame
        A pandas DataFrame of BIC values computed for all possible number of clusters
        given by range(min_components, max_components + 1) and all covariance
        structures given by covariance_type.
    ari_ : pandas.DataFrame
        Only computed when y is given. Pandas Dataframe containing ARI values computed
        for all possible number of clusters given by range(min_components,
        max_components) and all covariance structures given by covariance_type.
    """

    def __init__(
        self,
        min_components=2,
        max_components=None,
        covariance_type="full",
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

        if isinstance(covariance_type, np.ndarray):
            covariance_type = np.unique(covariance_type)
        elif isinstance(covariance_type, list):
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

        new_covariance_type = np.array(new_covariance_type)

        self.min_components = min_components
        self.max_components = max_components
        self.covariance_type = new_covariance_type
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
            np.array(bics),
            index=np.arange(lower_ncomponents, upper_ncomponents + 1),
            columns=self.covariance_type,
        )

        if y is not None:
            self.ari_ = pd.DataFrame(
                np.array(aris),
                index=np.arange(lower_ncomponents, upper_ncomponents + 1),
                columns=self.covariance_type,
            )
        else:
            self.ari_ = None

        # Finding the minimum bic for each covariance structure
        bic_mins = [min(bic) for bic in bics]
        bic_argmins = [np.argmin(bic) for bic in bics]

        # Find the index for the minimum bic amongst all covariance structures
        model_type_argmin = np.argmin(bic_mins)

        self.n_components_ = np.argmin(bics[model_type_argmin]) + 1
        self.model_ = models[model_type_argmin][bic_argmins[model_type_argmin]]

        return self
