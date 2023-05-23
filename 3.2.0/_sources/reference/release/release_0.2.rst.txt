Release Notes: GraSPy 0.2
=========================

We're happy to announce the release of GraSPy 0.2! GraSPy is a Python package for 
understanding the properties of random graphs that arise from modern datasets, such as
social networks and brain networks.

For more information, please visit our `website <http://graspy.neurodata.io/>`_
and our `tutorials <https://graspy.neurodata.io/tutorial.html>`_.


Highlights
----------
This release is the result of over 8 months of work with over 25 pull requests by 
10 contributors. Highlights include:

- Added ``AutoGMMCluster`` in ``cluster`` submodule. ``AutoGMMCluster`` is Python equivalent to ``mclust`` in R.
- Added ``subgraph`` submodule, which detects vertices that maximally correlates to given features.
- Added ``match`` submodule. Used for matching vertices from a pair of graphs with unknown vertex correspondence.
- Added functions for simulating a pair of correlated ER and SBM graphs.

Improvements
------------
- Diagonal augmentation is default behavior in AdjacencySpectralEmbed.
- Added functionality in ``to_laplacian`` to allow for directed graphs.
- Updated docstrings.
- Updated documentation website.
- Various bug fixes.

API Changes
-----------
- Added ``**kwargs`` argument for ``heatmap``.

Deprecations
------------
None

Contributors to this release
----------------------------
- `Jaewon Chung <https://github.com/j1c>`_
- `Benjamin Pedigo <https://github.com/bdpedigo>`_
- `Tommy Athey <https://github.com/tathey1>`_ (new contributor!)
- `Jayanta Dey <https://github.com/jdey4>`_ (new contributor!)
- `Iain Carmichael <https://github.com/idc9>`_ (new contributor!)
- `Shiyu Sun <https://github.com/shiyussy>`_ (new contributor!)
- `Ali Saad-Eldin <https://github.com/asaadeldin11>`_ (new contributor!)
- `Gun Kang <https://github.com/gkang7>`_ (new contributor!)
- `Shan Qiu <https://github.com/SHAAAAN>`_ (new contributor!)
- `Ben Falk <https://github.com/falkben>`_ (new contributor!)
- `Jennifer Heiko <https://github.com/jheiko1>`_ (new contributor!)