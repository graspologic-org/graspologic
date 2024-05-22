.. _ase_tutorial: https://microsoft.github.io/graspologic/tutorials/embedding/AdjacencySpectralEmbed.html

Embedding
=========

.. currentmodule:: graspologic.embed

Decomposition
-------------

.. autofunction:: select_dimension

.. autofunction:: select_svd

Single graph embedding
----------------------

.. autoclass:: AdjacencySpectralEmbed
.. autoclass:: LaplacianSpectralEmbed
.. autofunction:: node2vec_embed

Multiple graph embedding
------------------------

.. autoclass:: OmnibusEmbed
.. autoclass:: MultipleASE
.. autoclass:: mug2vec

Dissimilarity graph embedding
-----------------------------

.. autoclass:: ClassicalMDS
