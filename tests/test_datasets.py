# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import pytest
from graspologic.datasets import *


def test_drosphila_left():
    graph = load_drosophila_left()
    graph, labels = load_drosophila_left(return_labels=True)


def test_drosphila_right():
    graph = load_drosophila_right()
    graph, labels = load_drosophila_right(return_labels=True)


def test_load_mice():
    data = load_mice()
