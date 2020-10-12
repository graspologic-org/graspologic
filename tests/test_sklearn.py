# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import graspologic
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

# TODO: figure out a better solution here
# check_estimator(graspologic.embed.AdjacencySpectralEmbed)  # with current implementation of predict, this class is no longer sklearn compliant
check_estimator(graspologic.embed.LaplacianSpectralEmbed)
check_estimator(graspologic.embed.ClassicalMDS)
