..  -*- coding: utf-8 -*-

.. _contents:

Overview of GraSPy_
===================

.. _GraSPy: https://graspy.neurodata.io

GraSPy is a Python package for analysis of graphs, or networks. 

Motivation
----------

A graph, or network, provides a mathematically intuitive representation of data with some sort of relationship between items. For example, a social network can be represented as a graph by considering all participants in the social network as nodes, with connections representing whether each pair of individuals in the network are friends with one another. Naively, one might apply traditional statistical techniques to a graph, which neglects the spatial arrangement of nodes within the network and is not utilizing all of the information present in the graph. In this package, we provide utilities and algorithms designed for the processing and analysis of graphs with specialized graph statistical algorithms.

Python
------

Python is a powerful programming language that allows simple and flexible
representations of networks as well as clear and concise expressions of network
algorithms.  Python has a vibrant and growing ecosystem of packages that
NetworkX uses to provide more features such as numerical linear algebra and
drawing.  In order to make the most out of NetworkX you will want to know how
to write basic programs in Python.  Among the many guides to Python, we
recommend the `Python documentation <https://docs.python.org/3/>`_ and the text
by Alex Martelli [Martelli03]_.

Free software
-------------

GraSPy is free software; you can redistribute it and/or modify it under the
terms of the :doc:`Apache-2.0 </license>`.  We welcome contributions.
Join us on `GitHub <https://github.com/neurodata/graspy>`_.

History
-------

GraSPy was born in September 2018. The original version was designed and written by Jaewon Chung, Benjamin Pedigo, and Eric Bridgeford.

Documentation
=============

GraSPy is a graph statistics package in python.

.. toctree::
   :maxdepth: 1

   install
   tutorial
   reference/index
   news
   license
   bibliography



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`