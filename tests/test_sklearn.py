# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import graspologic
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

# check_estimator(graspologic.embed.AdjacencySpectralEmbed)  # not a valid sklearn estimator with oos predict method in current implementation
check_estimator(graspologic.embed.LaplacianSpectralEmbed)
check_estimator(graspologic.embed.ClassicalMDS)
