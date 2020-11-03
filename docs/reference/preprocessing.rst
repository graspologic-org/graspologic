Preprocessing
=============

.. currentmodule:: graspologic.preprocessing

Graph Cuts
----------

Constants
^^^^^^^^^
.. py:data:: LARGER_THAN_INCLUSIVE

Cut any edge or node > the ``cut_threshold``

.. py:data:: LARGER_THAN_EXCLUSIVE

Cut any edge or node >= the ``cut_threshold``

.. py:data:: SMALLER_THAN_INCLUSIVE

Cut any edge or node < the ``cut_threshold``

.. py:data:: SMALLER_THAN_EXCLUSIVE

Cut any edge or node <= the ``cut_threshold``

Classes
^^^^^^^
.. autoclass:: DefinedHistogram

Functions
^^^^^^^^^
.. autofunction:: cut_edges_by_weight

.. autofunction:: cut_vertices_by_betweenness_centrality

.. autofunction:: cut_vertices_by_degree_centrality

.. autofunction:: histogram_betweenness_centrality

.. autofunction:: histogram_degree_centrality

.. autofunction:: histogram_edge_weight
