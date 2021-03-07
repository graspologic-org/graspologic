.. _last-graspy-label:

Release Notes: GraSPy 0.3
=========================

We're happy to announce the release of GraSPy 0.3! GraSPy is a Python package for 
understanding the properties of random graphs that arise from modern datasets, such as
social networks and brain networks.

For more information, please visit our `website <http://graspy.neurodata.io/>`_
and our `tutorials <https://graspy.neurodata.io/tutorial.html>`_.


Highlights
----------
This release is the result of over 5 months of work with over 11 pull requests by 
7 contributors. Highlights include:

- Added seeded graph matching as a capability for graph matching, renamed graph matching class to ``GraphMatch`` 
- Added functions for simulating a pair of correlated RDPG graphs.
- Deprecated Python 3.5
- Added different backend hypothesis tests for the ``LatentDistributionTest`` from Hyppo
- Added a correction to make ``LatentDistributionTest`` valid for differently sized graphs

Improvements
------------
- Updated default value of ``rescale`` in RDPG simulation 
- Updated default value of ``scaled`` in MASE estimation 
- Improved error throwing in ``AutoGMM``
- Clarified the API for ``inference`` submodule

API Changes
-----------
- ``FastApproximateQAP`` was renamed to ``GraphMatch``
- ``fit`` method of ``LatentDistributionTest`` and ``LatentPositionTest`` now returns self instead of a p-value

Deprecations
------------
- Python 3.5

Contributors to this release
----------------------------
- `Jaewon Chung <https://github.com/j1c>`_
- `Benjamin Pedigo <https://github.com/bdpedigo>`_
- `Ali Saad-Eldin <https://github.com/asaadeldin11>`_
- `Shan Qiu <https://github.com/SHAAAAN>`_
- `Bijan Varjavand <https://github.com/bvarjavand>`_
- `Anton Alyakin <https://github.com/alyakin314>`_ (new contributor!)
- `Casey Weiner <https://github.com/caseypw>`_ (new contributor!)