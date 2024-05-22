Release Notes: GraSPy 0.1
=========================

We're happy to announce the release of GraSPy 0.1! GraSPy is a Python package for 
understanding the properties of random graphs that arise from modern datasets, such as
social networks and brain networks.

For more information, please visit our `website <http://graspy.neurodata.io/>`_
and our `tutorials <https://graspy.neurodata.io/tutorial.html>`_.


Highlights
----------
This release is the result of over 2 months of work with over 18 pull requests by 
3 contributors. Highlights include:

- Added ``MultipleASE``, which is a new method for embedding population of graphs.
- Added ``mug2vec`` within ``pipieline`` module, which learns a feature vector for population of graphs.

Improvements
------------
- Improved contribution guidelines.
- Fixed bugs in ``GaussianCluster``.
- ``symmeterize`` function now uses ``avg`` as default method.
- Fixed ``dataset`` module loading errors.
- Improve underlying `ER` sampling code.

API Changes
-----------
- Added ``sort_nodes`` argument for ``heatmap`` and ``gridplot`` functions.

Deprecations
------------
None

Contributors to this release
----------------------------
- `Jaewon Chung <https://github.com/j1c>`_
- `Benjamin Pedigo <https://github.com/bdpedigo>`_
- `Kiki Zhang <https://github.com/Kikiwink>`_ (new contributor!)