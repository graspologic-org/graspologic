# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import graspologic
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

# test
check_estimator(graspologic.embed.AdjacencySpectralEmbed)
check_estimator(graspologic.embed.LaplacianSpectralEmbed)
check_estimator(graspologic.embed.ClassicalMDS)
