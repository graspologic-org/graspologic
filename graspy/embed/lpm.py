#!/usr/bin/env python

# lpm.py
# Created by Eric Bridgeford on 2018-09-13.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np


class LatentPosition:
    """
	A basic class for a Latent Position Model.

    Attributes
    ----------
    X : array-like, shape (n_vertices, d)
        Estimated left vectors.
    Y : array-like, shape (n_vertices, d), or None
        Estimated right vectors. None if the input graph was undirected.
    d : array-like, shape(d, )
        Estimated singular values for each corresponding X or Y vectors.

    Notes
    -----
    Y only exists if the graphs that were embedded are directed.
	"""

    def __init__(self, X, Y, d=None):
        self.X = X
        if np.array_equal(X, Y):
            self.Y = None
        else:
            self.Y = Y
        self.d = d

    def is_symmetric(self):
        """
		A function to check whether a latent position model is symmetric.
		"""
        return (self.Y is None)

    def transform(self):
        """
        Computes the estimated latent positions.

        Returns
        -------
        out : array-like, shape (n_vertices, d), or tuple of array-like.
            If input graph was directed, then it returns both left and right
            estimated latent positions.
        """
        diagonal = np.diag(self.d**0.5)

        if self.Y is None:
            return np.dot(self.X, diagonal)
        else:
            return (np.dot(self.X, diagonal), np.dot(self.Y, diagonal))