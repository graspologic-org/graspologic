# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
# ruff: noqa: E402 SVD_SOLVER_TYPES needs to be first
"""
The embed module of ``graspologic.pipeline.embed`` is intended to provide faster
application development support. The functions provided in it reflect common call
patterns used when developing data processing pipelines and future consumption
by nearest neighbor services and visualization routines.
"""

__SVD_SOLVER_TYPES = ["randomized", "full", "truncated"]
from .adjacency_spectral_embedding import adjacency_spectral_embedding
from .embeddings import Embeddings, EmbeddingsView
from .laplacian_spectral_embedding import laplacian_spectral_embedding
from .omnibus_embedding import omnibus_embedding_pairwise
