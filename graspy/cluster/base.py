# omni.py
# Created by Jaewon Chung on 2018-10-04.
# Email: j1c@jhu.edu
# Copyright (c) 2018. All rights reserved.

from abc import abstractmethod

from sklearn.base import BaseEstimator, ClusterMixin


class BaseCluster(BaseEstimator, ClusterMixin):
    """
    Base clustering class.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

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

        return self