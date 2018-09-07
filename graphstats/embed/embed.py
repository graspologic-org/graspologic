#!/usr/bin/env python

# embed.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np
import networkx as nx
from abc import abstractmethod
from graphstats.utils import import_graph

class Embedding:
	"""
	A base class for embedding methods.
	"""

	def __init__(self, k=None):
		"""
		Inputs:
			k: int, optional (default None)
			 the desired number of embedding dimensions. If unspecified, uses
			 the optimal k as determined by graphstats.dimselect.
		"""
		self.k = k

	@abstractmethod
	def embed(self, graph):
		"""
		A method for embedding.

		Parameters:
		-----------
			graph: object

		Returns:
		--------
			X: array-like, shape (n_vertices, k)
				the estimated latent positions.
			Y: array-like, shape (n_vertices, k)
				if graph is not symmetric, the  right estimated latent
				positions. if graph is symmetric, "None".

		See Also:
			import_graph, sklearn.decomposition.TruncatedSVD
		"""