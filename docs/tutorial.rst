********
Tutorial
********

Simulations
===========
The following tutorials present several random graph models, such as Erdos-Renyi model, 
stochastic block model, and random dot product graph (RDPG) model. These models provide a basis for studying random graphs.

.. toctree::
   :maxdepth: 1
   
   tutorials/simulations/erdos_renyi
   tutorials/simulations/sbm
   tutorials/simulations/rdpg

Embedding
=========
Inference on random graphs depends on low-dimensional Euclidean representation of the vertices of graphs, known as *graph embeddings*, typically given by spectral decompositions of adjacency or Laplacian matrices. Below are tutorials for computing graph embeddings of single graph and multiple graphs.

.. toctree::
   :maxdepth: 1
   
   tutorials/embedding/AdjacencySpectralEmbed
   tutorials/embedding/Omnibus

Inference
===========================
Statistical testing on graphs requires specialized methodology in order to account 
for the fact that the edges and nodes of a graph are dependent on one another. Below 
are tutorials for robust statistical hypothesis testing on multiple graphs.

.. toctree::
   :maxdepth: 1

   tutorials/inference/semipar
   tutorials/inference/nonpar

Plotting
========
The following tutorials present ways to visualize the graphs, such as its adjacency matrix, and graph embeddings. 

.. toctree::
   :maxdepth: 1

   tutorials/plotting/heatmaps
   tutorials/plotting/gridplot
   tutorials/plotting/pairplot
