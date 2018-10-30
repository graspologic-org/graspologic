# base.py
# Created by Jaewon Chung on 2018-10-04.
# Email: j1c@jhu.edu
# Copyright (c) 2018. All rights reserved.

from abc import ABC, abstractmethod

from sklearn.base import BaseEstimator, ClusterMixin


class BaseCluster(ABC, BaseEstimator, ClusterMixin):
    """
    Base clustering class.
    """

    @abstractmethod
    def fit(self, X, y=None):
        """
        Compute clusters based on given method.

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

    @abstractmethod
    def predict(self, X, y=None):
        """
        Predict clusters based on best model.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        y : array-like, shape (n_samples, ), optional (default=None)
            List of labels for X if available. Used to compute
            ARI scores.

        Returns
        -------
        labels : array, shape (n_samples,)
            Component labels.

        ari : float
            Adjusted Rand index. Only returned if y is given.
        """

    @abstractmethod
    def fit_predict(self, X, y=None):
        """
        Fit the models and predict clusters based on best model.

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
