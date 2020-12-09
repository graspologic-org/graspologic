********
Tutorial
********

.. _models_tutorials:

Models
======
This tutorial presents several random graph models: the Erdos-Renyi (ER) model, degree-corrected ER model,
stochastic block model (SBM), degree-corrected SBM, and random dot product graph model. These models provide a basis for studying random graphs. All models are shown fit to the same dataset.

.. toctree::
   :maxdepth: 1

   tutorials/models/models

.. _simulations_tutorials:

Simulations
===========
The following tutorials demonstrate how to easily sample random graphs from graph models such as the Erdos-Renyi model,
stochastic block model, and random dot product graph (RDPG).

.. toctree::
   :maxdepth: 1

   tutorials/simulations/erdos_renyi
   tutorials/simulations/sbm
   tutorials/simulations/rdpg
   tutorials/simulations/corr
   tutorials/simulations/rdpg_corr

.. _cluster_tutorials:

Clustering
==========
The following tutorials explain how to cluster vertex or graph embeddings with two
clustering algorithms, as well as the advantages of these to comparable implementations.

.. toctree::
   :maxdepth: 1

   tutorials/clustering/autogmm
   tutorials/clustering/kclust

.. _embed_tutorials:

Embedding
=========
Inference on random graphs depends on low-dimensional Euclidean representation of the vertices of graphs, known as *graph embeddings*, typically given by spectral decompositions of adjacency or Laplacian matrices. Below are tutorials for computing graph embeddings of single graph and multiple graphs.

.. toctree::
   :maxdepth: 1

   tutorials/embedding/AdjacencySpectralEmbed
   tutorials/embedding/OutOfSampleEmbed
   tutorials/embedding/Omnibus
   tutorials/embedding/MASE

.. _inference_tutorials:

Inference
===========================
Statistical testing on graphs requires specialized methodology in order to account
for the fact that the edges and nodes of a graph are dependent on one another. Below
are tutorials for robust statistical hypothesis testing on multiple graphs.

.. toctree::
   :maxdepth: 1

   tutorials/inference/latent_position_test
   tutorials/inference/latent_distribution_test

.. _plot_tutorials:

Plotting
========
The following tutorials present ways to visualize the graphs, such as its adjacency matrix, and graph embeddings.

.. toctree::
   :maxdepth: 1

   tutorials/plotting/heatmaps
   tutorials/plotting/gridplot
   tutorials/plotting/pairplot
   tutorials/plotting/pairplot_with_gmm

.. _matching_tutorials:

Matching
========
The following tutorials demonstrate how to use the graph matching functionality,
including an introduction to the module, and how to utilize the seeding feature.

.. toctree::
   :maxdepth: 1

   tutorials/matching/faq
   tutorials/matching/sgm
   tutorials/matching/padded_gm

.. _subgraph_tutorials:

Subgraph
========
The following tutorial demonstrates how to estimate the signal-subgraph of samples of a graph/class model according to either the coherent or incoherent estimator models.

.. toctree::
   :maxdepth: 1

   tutorials/subgraph/subgraph

.. _aligning_tutorials:

Aligning
========
The following tutorials shows how to align two seperate datasets with each other, for better comparison of the data.

.. toctree::
   :maxdepth: 1

   tutorials/aligning/aligning
