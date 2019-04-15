import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted
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
        The minimum number of mixture components to consider.
        Must be less than max_components if max_components is not None

    max_components : int, defaults to None.
        The maximum number of mixture components to consider. 

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
            maximizes over ['spherical', 'diag', 'tied', 'full']

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
        Fitted GaussianMixture object fitted with optimal n_components.

    bic_ : list
        List of BIC values computed for all possible number of clusters
        given by range(1, max_components).

    ari_ : list
        Only computed when y is given. List of ARI values computed for 
        all possible number of clusters given by range(1, max_components).
    """

    def __init__(
        self,
        max_components=1,
        min_components=1,
        covariance_type="full",
        random_state=None,
    ):
        if isinstance(max_components, int):
            if max_components <= 0:
                msg = "max_components must be >= 1 or None."
                raise ValueError(msg)
            else:
                self.max_components = max_components
        else:
            msg = "max_components must be an integer, not {}.".format(
                type(max_components)
            )
            raise TypeError(msg)

        if isinstance(min_components, int):
            if min_components <= 0:
                msg = "min_components must be >= 1."
                raise ValueError(msg)
            elif min_components >= max_components:
                msg = "min_components must be strictly smaller than max_components"
        else:
            msg = "max_components must be an integer, not {}.".format(
                type(max_components)
            )
            raise TypeError(msg)

        self.max_components = max_components
        self.min_components = min_components
        self.covariance_type = covariance_type
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
        max_components = self.max_components
        min_components = self.min_components
        n_components = max_components - min_components + 1

        if max_components > X.shape[0]:
            msg = "n_components must be >= n_samples, but got \
                n_components = {}, n_samples = {}".format(
                self.max_components, X.shape[0]
            )
            raise ValueError(msg)

        # Get parameters
        random_state = self.random_state
        covariance_type = self.covariance_type

        if covariance_type == "all":
            covariances = ["spherical", "diag", "tied", "full"]
        elif type(covariance_type) is list:
            covariances = covariance_type
        else:
            covariances = [covariance_type]

        param_grid = dict(
            covariance_type=covariances,
            n_components=range(min_components, max_components + 1),
            random_state=[random_state],
        )

        param_grid = list(ParameterGrid(param_grid))

        models = [[] for _ in range(n_components)]
        bics = [[] for _ in range(n_components)]
        aris = [[] for _ in range(n_components)]

        for i, params in enumerate(param_grid):
            model = GaussianMixture(**params)
            model.fit(X)
            models[i % n_components].append(model)
            bics[i % n_components].append(model.bic(X))
            if y is not None:
                predictions = model.predict(X)
                aris[i % n_components].append(adjusted_rand_score(y, predictions))

        self.bic_ = pd.DataFrame(
            np.array(bics),
            index=np.arange(min_components, max_components + 1),
            columns=covariances,
        )

        if y is not None:
            self.ari_ = pd.DataFrame(
                np.array(aris),
                index=np.arange(min_components, max_components + 1),
                columns=covariances,
            )
        else:
            self.ari_ = None

        # Finding the minimum bic for each covariance structure
        bic_mins = [min(bic) for bic in bics]
        bic_argmins = [np.argmin(bic) for bic in bics]

        # Find the index for the minimum bic amongst all covariance structure
        model_type_argmin = np.argmin(bic_mins)

        self.n_components_ = np.argmin(bics[model_type_argmin]) + 1
        self.model_ = models[model_type_argmin][bic_argmins[model_type_argmin]]

        return self
