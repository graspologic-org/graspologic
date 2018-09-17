import numpy as np


class LatentPosition:
    """
	A basic class for a Latent Position Model.
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
