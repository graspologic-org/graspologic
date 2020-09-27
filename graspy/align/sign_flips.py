# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import numpy as np

from .base import BaseAlign


class SignFlips(BaseAlign):
    """
    This module is used to transform two datasets by flipping the signs of all
    entries along some dimensions. In particular, it does so in a way that
    brings these two datasets to the same orthant according to some criterion,
    computed along each dimension. The two critera currently available are the
    median and the maximum value in magnitude along each dimension.

    There are two choices for which orthant the datasets end up in: either the
    first orthant (i.e. with all criteras being positive), which is a default,
    or and orthant in which one the second dataset is already in.

    Parameters
    ----------
        criterion : string, {'median' (default), 'max'}, optional
            String describing the criterion used to choose whether to flip
            signs. Two options are currently supported:

            - 'median'
                Uses the median along each dimension
            - 'max'
                Uses the max (in magintude) alongs each dimension

    Attributes
    ----------
        Q_ : array, size (d, d)
            Final orthogonal matrix, used to modify `X`.

    """

    def __init__(
        self,
        criterion="median",
    ):
        # checking criterion argument
        if type(criterion) is not str:
            raise TypeError("Criterion must be str")
        if criterion not in ["median", "max"]:
            raise ValueError(f"{criterion} is not a valid criterion.")

        super().__init__()

        self.criterion = criterion

    def set_criterion_function(self):
        # perform a check, in case it was modified directly
        if self.criterion not in ["median", "max"]:
            raise ValueError(f"{self.criterion} is not a valid criterion")

        if self.criterion == "median":

            def median_criterion(X):
                return np.median(X, axis=0)

            self.criterion_function_ = median_criterion
        if self.criterion == "max":

            def max_criterion(X):
                return X[np.argmax(np.abs(X), axis=0), np.arange(X.shape[1])]

            self.criterion_function_ = max_criterion

    def fit(self, X, Y):
        """
        Uses the two datasets to learn the matrix `self.Q_` that aligns the
        first dataset with the second.

        In sign flips, `self.Q_` is an diagonal orthogonal matrices (i.e. a
        matrix with 1 or -1 in each entry on diagonal and 0 everywhere else)
        picked such that all dimensions of `X` @ `self.Q_` and `Y` are in the
        same orthant using some critera (median or max magnitude).

        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            First dataset of vectors. These vectors need to have same number of
            dimensions as ones in `Y`, but the number of vectors can differ.

        Y : np.ndarray, shape (m, d)
            Second dataset of vectors. These vectors need to have same number
            of dimensions as ones in `X`, but the number of vectors can differ.

        Returns
        -------
        self : returns an instance of self
        """
        X, Y = self._check_datasets(X, Y)
        _, d = X.shape

        self.set_criterion_function()
        X_criterias = self.criterion_function_(X)
        Y_criterias = self.criterion_function_(Y)

        val = np.multiply(X_criterias, Y_criterias)
        t_X = (val >= 0) * 2 - 1

        self.Q_ = np.diag(t_X)
        return self
