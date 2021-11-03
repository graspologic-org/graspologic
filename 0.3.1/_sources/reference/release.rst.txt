..  -*- coding: utf-8 -*-

Release Log
===========

graspologic 0.3.0
-----------------
- Fixed imports for hyppo >= 0.2.0
  `#785 <https://github.com/microsoft/graspologic/pull/785>`_
- Added ``trials`` parameter to leiden and set a new requirement of
  ``graspologic-native>=1.0.0``
  `#790 <https://github.com/microsoft/graspologic/pull/790>`_
- Updated lcc check in utils to use ``scipy's`` connected_components in
  ``is_fully_connected``
  `#708 <https://github.com/microsoft/graspologic/pull/708>`_
- Added Out of Sample Laplacian Spectral Embedding
  `#722 <https://github.com/microsoft/graspologic/pull/722>`_
- Added parallelization of sampling process in Latent Distribution Test
  `#744 <https://github.com/microsoft/graspologic/pull/744>`_
- Added Covariate Assisted Spectral Embedding
  `#599 <https://github.com/microsoft/graspologic/pull/599>`_
- Added sparse matrix support for some matrix plotting functions
  `#794 <https://github.com/microsoft/graspologic/pull/794>`_
- Add ``scipy's`` connected component finder for sparse support in
  ``largest_connected_component``
  `#795 <https://github.com/microsoft/graspologic/pull/795>`_
- Default weight to 1 for all edges in unweighted graph for node2vec
  `#789 <https://github.com/microsoft/graspologic/pull/789>`_
- Fixed a bug with 0's in sparse matrix for largest connected component calculation
  `#805 <https://github.com/microsoft/graspologic/pull/805>`_
- Usage of the 'un-bearably' awesome `beartype <https://github.com/beartype/beartype>`_
  library for type checking `#819 <https://github.com/microsoft/graspologic/pull/819>`_
- Added directed graph support to automated layouts
  `#807 <https://github.com/microsoft/graspologic/pull/807>`_
- Fixed bug in mug2vec around ``pass_to_ranks``
  `#821 <https://github.com/microsoft/graspologic/pull/821>`_
- Pipeline module released, which includes a ``networkx`` based API for using
  Adjacency or Laplacian Spectral Embeddings and the Omnibus Spectral Embedding.
  `#814 <https://github.com/microsoft/graspologic/pull/814>`_,
  `#817 <https://github.com/microsoft/graspologic/pull/817>`_,
  `#823 <https://github.com/microsoft/graspologic/pull/823>`_,
  `#824 <https://github.com/microsoft/graspologic/pull/824>`_
- Add option for more than one kmeans init to autogmm
  `#662 <https://github.com/microsoft/graspologic/pull/662>`_
- Added sparse support for Omnibus embeddings
  `#834 <https://github.com/microsoft/graspologic/pull/834>`_
- Added LSE as an embedding for use within the Omnibus embeddings
  `#835 <https://github.com/microsoft/graspologic/pull/835>`_
- Clarified behavior of Leiden for graphs with isolates
  `#830 <https://github.com/microsoft/graspologic/pull/830>`_
- Updated ``utils.is_unweighted`` to use more efficient scipy count and filter methods
  for CSR matrices `#836 <https://github.com/microsoft/graspologic/pull/836>`_
- Updated default values for node2vec parameters to be more in line with most common
  production settings `#838 <https://github.com/microsoft/graspologic/pull/838>`_

graspologic 0.2.0
-----------------
- Documentation fixes and updates
  `#729 <https://github.com/microsoft/graspologic/pull/729>`_ and
  `#743 <https://github.com/microsoft/graspologic/pull/743>`_
- Fixed `custom initialization handling for GraphMatch
  #737 <https://github.com/microsoft/graspologic/pull/737>`_
- Fixed `incorrect use of optimal transport params in seedless procrustes
  #745 <https://github.com/microsoft/graspologic/pull/745>`_
- Added `Similarity Augmented Graph Matching #560
  <https://github.com/microsoft/graspologic/pull/560>`_
- Added `parallelized boostrapping to the latent position test #710
  <https://github.com/microsoft/graspologic/pull/710>`_
- Fixed `outlier handling in quadratic assignment #754
  <https://github.com/microsoft/graspologic/pull/754>`_
- Fixed GraphMatch random_state behavior when parallelizing in `#770
  <https://github.com/microsoft/graspologic/pull/770>`_
- Modified autolayout to allow you to opt out of the slow no overlap / occlusion
  removal algorithm
- Fixed bug in leiden where graphs with non-string node IDs were being returned as their
  ``str()`` representations instead of their original values.
- Fixed a number of bugs in autolayout
- Fixed multiple bugs in MASE,
  `#766 <https://github.com/microsoft/graspologic/pull/766>`_ and
  `#768 <https://github.com/microsoft/graspologic/pull/768>`_
- Fixed bug in node2vec where non-string node IDs were always being returned as their
  ``str()`` representation in the labels array
- Fixed bug in autolayout where non-string node IDs broke (see: leiden fix and node2vec
  fix)


graspologic 0.1.0
-----------------
This release represents the first release of the ``GraSPy`` and `topologic`_ libraries
merger!

In addition to all of the features that ``GraSPy`` had in :ref:`last-graspy-label`,
this release includes a number of feature enhancements and bug fixes - frankly, too
many to go into.

Please treat this as a brand new first release, informed heavily by the prior
``GraSPy``, and extra special thanks to **all** of our
`contributors <https://github.com/microsoft/graspologic/blob/dev/CONTRIBUTORS.md>`_!

Previous GraSPy Releases
------------------------
.. toctree::
   :maxdepth: 1

   release/graspy_releases.rst

.. _topologic: https://github.com/microsoft/topologic
