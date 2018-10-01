from sklearn.base import BaseEstimator, ClusterMixin


class BaseCluster(BaseEstimator, ClusterMixin):
    """
    Base clustering class.
    Parameters
    ----------
    
    """

    def __init__(self, method, *args, **kwargs):
        self.method = method(*args, **kwargs)

    @abstractmethod
    def fit(self, X):
        """
        Compute clusters based on given method.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        Returns
        -------
        self
        """
        # call self._reduce_dim(A) from your respective embedding technique.
        # import graph(s) to an adjacency matrix using import_graph function
        # here

        return self