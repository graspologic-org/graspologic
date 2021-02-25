Release Notes: GraSPy 0.0.3
===========================

We're happy to announce the release of GraSPy 0.0.3! GraSPy is a Python package for 
understanding the properties of random graphs that arise from modern datasets, such as
social networks and brain networks.

For more information, please visit our `website <http://graspy.neurodata.io/>`_
and our `tutorials <https://graspy.neurodata.io/tutorial.html>`_.


Highlights
----------
This release is the result of over 2 months of work with over 16 pull requests by 
4 contributors. Highlights include:

- Optimization over covariance structures when using ``GaussianCluster``
- Standardized sorting for visualizing graphs when using ``heatmap`` or ``gridplot``
- Graph model classes for fitting several random graph models to input datasets
- Improved customization for ``heatmaps`` and ``gridplots``


Improvements
------------
- Added badges to Github for arxiv paper and number of downloads
- Remove author headers for individual source files 
- Fix bugs in documentation
- Bug fix for calculating intersection of largest connected components between graphs
- Pre-defined axes can be passed to ``heatmap`` for making subplot figures
- Colormap objects and color bounds can be passed to ``heatmap`` directly

API Changes
-----------
- ``SemiparametricTest`` was renamed to ``LatentPositionTest``
- ``NonparametricTest`` was renamed to ``LatentDistributionTest``
- ``heatmap`` and ``gridplot`` accept ``hier_label_fontsize`` and ``title_pad`` kwargs

Deprecations
------------
- The notebooks folder was removed from ``GraSPy``
- ``SemiparametricTest`` and ``NonparametricTest`` renamed (see above)

Contributors to this release
----------------------------
- `Benjamin Pedigo <https://github.com/bdpedigo>`_
- `Jaewon Chung <https://github.com/j1c>`_
- `Hayden Helm <https://github.com/hhelm10>`_ (new contributor!)
- `Alex Loftus <https://github.com/loftusa>`_ (new contributor!)