..  -*- coding: utf-8 -*-

Release Log
===========

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
