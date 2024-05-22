Release Notes: GraSPy 0.0.2
===========================

We're happy to announce the release of GraSPy 0.0.2! GraSPy is a Python package for 
understanding the properties of random graphs that arise from modern datasets, such as social networks 
and brain networks.

For more information, please visit our `website <http://graspy.neurodata.io/>`_
and our `tutorials <https://graspy.neurodata.io/tutorial.html>`_.


Highlights
----------
This release is the result of 3 months of work with over 16 pull requests by 5 contributors. Highlights include:

- Nonparametric hypothesis testing method for testing two non-vertex matched graphs.
- Plotting updates to ``pairplot``, ``gridplot`` and ``heatmaps``.
- Sampling degree-correlcted stochatic block models (DC-SBM).
- ``import_edgelist`` function for importing single or multiple edgelists.
- Enforcing ``Black`` formatting for the package.

Improvements
------------
- Embedding methods are now fully sklearn-compliant. This is tested via ``check_estimator`` function in sklearn.
- ``gridplot`` and ``heatmap`` can now plot hierchical labels.
- New Laplacian computing method ('R-DAD') by adding a constant to the diagonal degree matrix.
- Semiparametric testing only checks for largest connected component (LCC) in the intial embeddings. 
- Various bug fixes.
- Various tutorial latex fixes.
- Various documentation clarifications.
- More consistent documentation.

API Changes
-----------
- ``check_lcc`` argument in ``AdjacencySpectralEmbed``, ``LaplacianSpectralEmbed``, and ``OmnibusEmbed`` classes, which checks if input graph(s) are fully connected when ``check_lcc`` is True.
- ``gridplot`` and ``heatmap`` now have a ``inner_hier_labels`` and ``outer_hier_labels``, which are used for hierarchical labeling of nodes.
- ``to_laplacian`` function now has ``regularizer`` arg for when ``form`` is 'R-DAD'.
- ``sbm`` function now has ``dc`` and ``dc_kws`` arguments for sampling SBM with degree-correction.

Deprecations
------------
None.

Contributors to this release
----------------------------
- `Benjamin Pedigo <https://github.com/bdpedigo>`_
- `Jaewon Chung <https://github.com/j1c>`_
- `Bijan Varjavand <https://github.com/bvarjavand>`_
- `Vikram Chandrashekhar <https://github.com/vikramc1>`_
- `Ronan Perry <https://github.com/rflperry>`_
