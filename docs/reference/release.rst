..  -*- coding: utf-8 -*-

Release Log
===========

graspologic 3.3.0
-----------------
- Added features and bugfixes to ``heatmap``
  `#750 <https://github.com/graspologic-org/graspologic/pull/750>`
- Fixed type specification bugs related to Numpy 1.25 release
  `#1047 <https://github.com/graspologic-org/graspologic/pull/1047>`
- Added option for more efficient graph matching matrix operations
  `#1046 <https://github.com/graspologic-org/graspologic/pull/1046>`
- Added an axis argument to ``screeplot``
  `#1048 <https://github.com/graspologic-org/graspologic/pull/1048>`
- Fixed compatibility issues related to matplotlib 3.8 release 
  `#1049 <https://github.com/graspologic-org/graspologic/pull/1049>`

graspologic 3.2.0
-----------------
- Added Python 3.11 support
  `#1039 <https://github.com/graspologic-org/graspologic/pull/1039>`

graspologic 3.1.0
-----------------
- Added density and group connection tests
  `#1032 <https://github.com/graspologic-org/graspologic/pull/1032>`
- Fixed NetworkX 3 compatibility and switched to sparse arrays
  `#1018 <https://github.com/graspologic-org/graspologic/commit/13d0d466fd1f28c6504b83ae63c16e53c4445997>`

graspologic 3.0.0
-----------------
- Corrected contributing guidelines
  `#1014 <https://github.com/graspologic-org/graspologic/pull/1014>`
- Corrected deprecation warnings
  `#1019 <https://github.com/graspologic-org/graspologic/pull/1019>`
- Limited NetworkX version due to incompatibility
  `#1016 <https://github.com/graspologic-org/graspologic/pull/1016>`
- Added Python 3.10 and removed Python 3.7 support
  `#1010 <https://github.com/graspologic-org/graspologic/pull/1010>`

graspologic 2.0.1
-----------------
- Fixed bug with a matplotlib version incompatibility
  `#996 <https://github.com/graspologic-org/graspologic/pull/996>`
- Fixed graph matching with similarity matrix of unequal dimensions
  `#1002 <https://github.com/graspologic-org/graspologic/pull/1002>`
- Fixed bug with missing typing-extensions dependency
  `#999 <https://github.com/graspologic-org/graspologic/pull/999>`

graspologic 2.0.0
-----------------
- Refactored graph matching code and added many new features
  `#960 <https://github.com/graspologic-org/graspologic/pull/960>`
- Added elbow marker to screeplot in plot module
  `#979 <https://github.com/graspologic-org/graspologic/pull/979>`
- Fixed mug2vec behavior for directed graphs
  `#968 <https://github.com/graspologic-org/graspologic/pull/968>`
- Fixed typo in aligning tutorial
  `#974 <https://github.com/graspologic-org/graspologic/pull/974>`
- Added sex labels to mice dataset
  `#967 <https://github.com/graspologic-org/graspologic/pull/967>`
- Made improvements to contributing guidelines
  `#973 <https://github.com/graspologic-org/graspologic/pull/973>`
- Corrected notation in documentation of to_laplacians
  `#969 <https://github.com/graspologic-org/graspologic/pull/969>`
- Fixed isolated nodes handling in node2vec
  `#953 <https://github.com/graspologic-org/graspologic/pull/953>`
- Fixed repeated numba compilation in EdgeSwapper
  `#965 <https://github.com/graspologic-org/graspologic/pull/965>`
- Fixed intersphinx bug
  `#963 <https://github.com/graspologic-org/graspologic/pull/963>`
- Removed default axis labels in networkplot
  `#954 <https://github.com/graspologic-org/graspologic/pull/954>`
- Fixed reproducibility in EdgeSwapper and added to docs
  `#945 <https://github.com/graspologic-org/graspologic/pull/945>` 
- Added Degree Preserving Edge Swaps
  `#935 <https://github.com/graspologic-org/graspologic/pull/935>`
- Fixed mypy issue
  `#943 <https://github.com/graspologic-org/graspologic/pull/943>`
- Fixed loops bug in SBM and DCSBM model fitting
  `#930 <https://github.com/graspologic-org/graspologic/pull/930>` 
- Added error message in Leiden when given a multigraph was incorrect
  `#926 <https://github.com/graspologic-org/graspologic/pull/926>`
- Fixed typos in ER and SBM models
  `#920 <https://github.com/graspologic-org/graspologic/pull/920>`

graspologic 1.0.0
-----------------
- Removed Python 3.6 support
- Officially added Python 3.9 support
- Fixed a type in an error message 
  `#904 <https://github.com/graspologic-org/graspologic/pull/904>`
- Added support for arbitrarily indexed node data for networkplot 
  `#906 <https://github.com/graspologic-org/graspologic/pull/906>`
- Fixed compatibility issues with gensim, black, mypy, and 
  setup-python `#913 <https://github.com/graspologic-org/graspologic/pull/913>`
- Fixed a bug in leiden/hierarchical_leiden 
  `#902 <https://github.com/graspologic-org/graspologic/pull/902>`
- Ensured reproducibility in latent_distribution_test
  `#892 <https://github.com/graspologic-org/graspologic/pull/892>`
- Fixed documentation in KMeansCluster
  `#892 <https://github.com/graspologic-org/graspologic/pull/829>`
- Fixed bug in automatic layouts
  `#894 <https://github.com/graspologic-org/graspologic/pull/894>`
- Added Python 3.9 support
  `#889 <https://github.com/graspologic-org/graspologic/pull/889>`
- Added type hinting
  `#543 <https://github.com/graspologic-org/graspologic/pull/543>`
- Modified order of returns in inference module
  `#859 <https://github.com/graspologic-org/graspologic/pull/859>`
- Changes to documentation publish pipeline
  `#874 <https://github.com/graspologic-org/graspologic/pull/874>`
- Added function networkplot for drawing 2D networks
  `#860 <https://github.com/graspologic-org/graspologic/pull/860>`
- Removed dtype requirement in vertex nomination 
  `#865 <https://github.com/graspologic-org/graspologic/pull/865>`


graspologic 0.3.0
-----------------
- Fixed imports for hyppo >= 0.2.0
  `#785 <https://github.com/graspologic-org/graspologic/pull/785>`_
- Added ``trials`` parameter to leiden and set a new requirement of
  ``graspologic-native>=1.0.0``
  `#790 <https://github.com/graspologic-org/graspologic/pull/790>`_
- Updated lcc check in utils to use ``scipy's`` connected_components in
  ``is_fully_connected``
  `#708 <https://github.com/graspologic-org/graspologic/pull/708>`_
- Added Out of Sample Laplacian Spectral Embedding
  `#722 <https://github.com/graspologic-org/graspologic/pull/722>`_
- Added parallelization of sampling process in Latent Distribution Test
  `#744 <https://github.com/graspologic-org/graspologic/pull/744>`_
- Added Covariate Assisted Spectral Embedding
  `#599 <https://github.com/graspologic-org/graspologic/pull/599>`_
- Added sparse matrix support for some matrix plotting functions
  `#794 <https://github.com/graspologic-org/graspologic/pull/794>`_
- Add ``scipy's`` connected component finder for sparse support in
  ``largest_connected_component``
  `#795 <https://github.com/graspologic-org/graspologic/pull/795>`_
- Default weight to 1 for all edges in unweighted graph for node2vec
  `#789 <https://github.com/graspologic-org/graspologic/pull/789>`_
- Fixed a bug with 0's in sparse matrix for largest connected component calculation
  `#805 <https://github.com/graspologic-org/graspologic/pull/805>`_
- Usage of the 'un-bearably' awesome `beartype <https://github.com/beartype/beartype>`_
  library for type checking `#819 <https://github.com/graspologic-org/graspologic/pull/819>`_
- Added directed graph support to automated layouts
  `#807 <https://github.com/graspologic-org/graspologic/pull/807>`_
- Fixed bug in mug2vec around ``pass_to_ranks``
  `#821 <https://github.com/graspologic-org/graspologic/pull/821>`_
- Pipeline module released, which includes a ``networkx`` based API for using
  Adjacency or Laplacian Spectral Embeddings and the Omnibus Spectral Embedding.
  `#814 <https://github.com/graspologic-org/graspologic/pull/814>`_,
  `#817 <https://github.com/graspologic-org/graspologic/pull/817>`_,
  `#823 <https://github.com/graspologic-org/graspologic/pull/823>`_,
  `#824 <https://github.com/graspologic-org/graspologic/pull/824>`_
- Add option for more than one kmeans init to autogmm
  `#662 <https://github.com/graspologic-org/graspologic/pull/662>`_
- Added sparse support for Omnibus embeddings
  `#834 <https://github.com/graspologic-org/graspologic/pull/834>`_
- Added LSE as an embedding for use within the Omnibus embeddings
  `#835 <https://github.com/graspologic-org/graspologic/pull/835>`_
- Clarified behavior of Leiden for graphs with isolates
  `#830 <https://github.com/graspologic-org/graspologic/pull/830>`_
- Updated ``utils.is_unweighted`` to use more efficient scipy count and filter methods
  for CSR matrices `#836 <https://github.com/graspologic-org/graspologic/pull/836>`_
- Updated default values for node2vec parameters to be more in line with most common
  production settings `#838 <https://github.com/graspologic-org/graspologic/pull/838>`_

graspologic 0.2.0
-----------------
- Documentation fixes and updates
  `#729 <https://github.com/graspologic-org/graspologic/pull/729>`_ and
  `#743 <https://github.com/graspologic-org/graspologic/pull/743>`_
- Fixed `custom initialization handling for GraphMatch
  #737 <https://github.com/graspologic-org/graspologic/pull/737>`_
- Fixed `incorrect use of optimal transport params in seedless procrustes
  #745 <https://github.com/graspologic-org/graspologic/pull/745>`_
- Added `Similarity Augmented Graph Matching #560
  <https://github.com/graspologic-org/graspologic/pull/560>`_
- Added `parallelized boostrapping to the latent position test #710
  <https://github.com/graspologic-org/graspologic/pull/710>`_
- Fixed `outlier handling in quadratic assignment #754
  <https://github.com/graspologic-org/graspologic/pull/754>`_
- Fixed GraphMatch random_state behavior when parallelizing in `#770
  <https://github.com/graspologic-org/graspologic/pull/770>`_
- Modified autolayout to allow you to opt out of the slow no overlap / occlusion
  removal algorithm
- Fixed bug in leiden where graphs with non-string node IDs were being returned as their
  ``str()`` representations instead of their original values.
- Fixed a number of bugs in autolayout
- Fixed multiple bugs in MASE,
  `#766 <https://github.com/graspologic-org/graspologic/pull/766>`_ and
  `#768 <https://github.com/graspologic-org/graspologic/pull/768>`_
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
`contributors <https://github.com/graspologic-org/graspologic/blob/dev/CONTRIBUTORS.md>`_!

Previous GraSPy Releases
------------------------
.. toctree::
   :maxdepth: 1

   release/graspy_releases.rst

.. _topologic: https://github.com/microsoft/topologic
