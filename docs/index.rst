..  -*- coding: utf-8 -*-

.. _contents:

Overview of graspologic_
========================

.. _graspologic: https://graspologic.readthedocs.org/en/latest

graspologic is a Python package for analysis of graphs, or networks.

Motivation
----------

A graph, or network, provides a mathematically intuitive representation of data with
some sort of relationship between items. For example, a social network can be
represented as a graph by considering all participants in the social network as nodes,
with connections representing whether each pair of individuals in the network are friends
with one another. Naively, one might apply traditional statistical techniques to a graph,
which neglects the spatial arrangement of nodes within the network and is not utilizing
all of the information present in the graph. In this package, we provide utilities and
algorithms designed for the processing and analysis of graphs with specialized graph
statistical algorithms.

Python
------

Python is a powerful programming language that allows concise expressions of network
algorithms.  Python has a vibrant and growing ecosystem of packages that
graspologic uses to provide more features such as numerical linear algebra and
plotting.  In order to make the most out of graspologic you will want to know how
to write basic programs in Python.  Among the many guides to Python, we
recommend the `Python documentation <https://docs.python.org/3/>`_.

Free software
-------------

graspologic is free software; you can redistribute it and/or modify it under the
terms of the :doc:`MIT </license>` license.  We welcome contributions.
Join us on `GitHub <https://github.com/microsoft/graspologic>`_.

History
-------

``graspologic`` first released in September 2020, but it got its start as a pair of Python libraries
written by Johns Hopkins University's NeuroData lab and Microsoft Research's Project Essex.
Both teams worked on many of the same algorithms, shared research, findings, and generally duplicated a lot of effort.

``GraSPy`` - the NeuroData library - and ``topologic`` - the Microsoft Research library began merging in September of 2020, but both got their starts far earlier, with GraSPy starting in September 2018 and topologic starting just a short time later, on October 2nd, 2018.

GraSPy was originally designed and written by Jaewon Chung, Benjamin Pedigo, and Eric Bridgeford.

Topologic was originally designed and written by Patrick Bourke, Jonathan McLean, Nick Caurvina, and Dwayne Pryce.

.. toctree-filt::
   :maxdepth: 1
   :caption: Documentation

   license
   reference/index
   tutorials/index

.. toctree::
   :maxdepth: 1
   :caption: Useful Links

   graspologic @ GitHub <http://www.github.com/microsoft/graspologic/>
   graspologic @ PyPI <https://pypi.org/project/graspologic/>
   Issue Tracker <https://github.com/microsoft/graspologic/issues>

Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
