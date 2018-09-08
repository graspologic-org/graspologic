#!/usr/bin/env python

# embed.py
# Created by Eric Bridgeford on 2018-09-07.
# Email: ebridge2@jhu.edu
# Copyright (c) 2018. All rights reserved.

import numpy as np
import networkx as nx
from abc import abstractmethod
from graphstats.utils import import_graph
from sklearn.decomposition import TruncatedSVD
from graphstats.embed import selectSVD


class Embedding:
	"""
	A base class for embedding methods.
	"""

	def __init__(self, method=selectSVD, *args, **kwargs):
		"""
		A class for embedding a graph.
		
		Inputs:
		-------
			method: object (default selectSVD)
			args: list, optional (default None)
			 options taken by the desired embedding method as arguments.
			kwargs: dict, optional (default None)
			 options taken by the desired embedding method as key-worded
			 arguments.

		See Also:
		---------
			graphstats.select.dim
		"""
		self.method=method
		self.args = args
		self.kwargs = kwargs

	def reduce_dim(self, A):
		"""
		A function that reduces the dimensionality of an adjacency matrix
		using the desired embedding method.

		Parameters:
		-----------
			A: {array-like}, shape (n_vertices, n_vertices)
			 the adjacency matrix to embed.
		"""
		self.method(A, *args, **kwargs)

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
			import_graph
		"""