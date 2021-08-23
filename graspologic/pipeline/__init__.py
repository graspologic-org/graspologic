"""
The ``pipeline`` module includes a collection of higher level API abstractions from
the functionality exposed elsewhere in ``graspologic``. The classes, functions, and
modules elsewhere in ``graspologic`` are intended to provide fine-grained, expert-level
control over the features they implement. These building blocks provide an excellent
backbone of utility, for researchers in mathematics and science, especially as they
hew so closely to ``scikit-learn``'s programming paradigms and object model.

But for software engineers and datascientists, there is a certain ritualistic cost to
preparing a graph, setting up the objects for use, and tearing them down afterwards.

``pipeline`` is intended to smooth the transition between a common developer and
a graph machine learning subject matter expert. We make a presumption that most
programmers are software developers first, and dabbling in ML second, and our intention
is to bridge this gap.

"""
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from . import embed
from .graph_builder import GraphBuilder
