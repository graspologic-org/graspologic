# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

from abc import abstractmethod
from sklearn.base import BaseEstimator


class BaseVN(BaseEstimator):
    def __init__(self, multigraph=False):
        self.multigraph = multigraph

    @abstractmethod
    def fit(self, X, y):
        return None

    @abstractmethod
    def predict(self):
        return None

    @abstractmethod
    def fit_transform(self, X, y):
        return None
