import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.utils.validation import check_is_fitted

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
    max_components : int, defaults to 1.
        The maximum number of mixture components to consider.

    covariance_type : {'full' (default), 'tied', 'diag', 'spherical'}, optional
        String describing the type of covariance parameters to use.
        Must be one of:

        - 'full'
            each component has its own general covariance matrix
        - 'tied'
            all components share the same general covariance matrix
        - 'diag'
            each component has its own diagonal covariance matrix
        - 'spherical'
            each component has its own single variance
    
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

    def __init__(self,
                 max_components=1,
                 covariance_type='full',
                 random_state=None):
        if isinstance(max_components, int):
            if max_components <= 0:
                msg = "n_components must be >= 1 or None."
                raise ValueError(msg)
            else:
                self.max_components = max_components
        else:
            msg = 'max_components must be an integer, not {}.'.format(
                type(max_components))
            raise TypeError(msg)
        self.max_components = max_components
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
        if max_components > X.shape[0]:
            msg = "n_components must be >= n_samples, but got \
                n_components = {}, n_samples = {}".format(
                self.max_components, X.shape[0])
            raise ValueError(msg)

        # Get parameters
        random_state = self.random_state
        covariance_type = self.covariance_type

        # Compute all models
        models = []
        bics = []
        aris = []
        for n in range(1, max_components + 1):
            model = GaussianMixture(
                n_components=n,
                covariance_type=covariance_type,
                random_state=random_state)

            # Fit and compute values
            model.fit(X)
            models.append(model)
            bics.append(model.bic(X))
            if y is not None:
                predictions = model.predict(X)
                aris.append(adjusted_rand_score(y, predictions))

        self.bic_ = bics
        if y is not None:
            self.ari_ = aris
        else:
            self.ari_ = None
        self.n_components_ = np.argmin(bics) + 1
        self.model_ = models[np.argmin(bics)]

        return self
