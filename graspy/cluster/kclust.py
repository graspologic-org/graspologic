import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.utils.validation import check_is_fitted

from .base import BaseCluster


class KMeansCluster(BaseCluster):
    """
    KMeans

    Representation of a Gaussian mixture model probability distribution. 
    This class allows to estimate the parameters of a Gaussian mixture 
    distribution. It computes all possible models from one component to 
    max_clusters. The best model is given by the lowest BIC score.

    Parameters
    ----------
    max_clusters : int, defaults to 1.
        The maximum number of mixture components to consider.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.

    Attributes
    ----------
    n_clusters_ : int
        Optimal number of components. If y is given, it is based on largest 
        ARI. Otherwise, it is based on smallest loss.
    
    model_ : KMeans object
        Fitted KMeans object fitted with optimal n_components.

    losses_ : list
        List of mean squared error values computed for all possible number 
        of clusters given by range(1, max_clusters).

    ari_ : list
        Only computed when y is given. List of ARI values computed for 
        all possible number of clusters given by range(1, max_clusters).
    """

    def __init__(self, max_clusters=1, random_state=None):
        self.max_clusters = max_clusters
        self.random_state = random_state

    def fit(self, X, y=None):
        """
        Fits kmeans model to the data. 

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        
        y : array-like, shape (n_samples,), optional (default=None)
            List of labels for X if available. Used to compute ARI scores.

        Returns
        -------
        self
        """
        # Deal with number of clusters
        if self.max_clusters <= 0:
            msg = "n_components must be >= 1 or None."
            raise ValueError(msg)
        elif self.max_clusters > X.shape[0]:
            msg = "n_components must be >= n_samples, but got \
                n_components = {}, n_samples = {}".format(
                self.max_clusters, X.shape[0])
            raise ValueError(msg)
        elif self.max_clusters >= 1:
            max_clusters = self.max_clusters
        elif self.max_clusters is None:
            max_clusters = 1

        # Get parameters
        random_state = self.random_state

        # Compute all models
        models = []
        losses = []
        aris = []
        for n in range(1, max_clusters + 1):
            model = KMeans(n_clusters=n, random_state=random_state)

            # Fit and compute values
            model.fit(X)
            models.append(model)
            losses.append(model.inertia_)
            if y is not None:
                predictions = model.predict(X)
                aris.append(adjusted_rand_score(y, predictions))

        if y is not None:
            self.ari_ = aris
            self.n_clusters_ = np.argmax(aris) + 1
            self.model_ = models[np.argmax(aris)]
        else:
            self.ari_ = None
            self.n_clusters_ = np.argmin(losses) + 1
            self.model_ = models[np.argmin(losses)]

        return self

    def fit_predict(self, X, y=None):
        """
        Estimate model parameters using X and predict the labels for X
        using the model given by best BIC score.

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
        labels : array, shape (n_samples,)
            Component labels.

        ari : float
            Adjusted Rand index. Only returned if y is given.
        """
        self.fit(X, y)

        if y is None:
            labels = self.predict(X, y)
            return labels
        else:
            labels, ari = self.predict(X, y)
            return labels, ari

    def predict(self, X, y=None):
        """
        Predict the labels for X using the model given by best BIC score.

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
        labels : array, shape (n_samples,)
            Component labels.

        ari : float
            Adjusted Rand index. Only returned if y is given.
        """
        # Check if fit is already called
        check_is_fitted(self, ['model_'], all_or_any=all)
        labels = self.model_.predict(X)

        if y is None:
            return labels
        else:
            ari = adjusted_rand_score(y, labels)
            return labels, ari