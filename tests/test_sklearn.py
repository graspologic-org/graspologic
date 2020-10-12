# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import graspologic
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

<<<<<<< HEAD
check_estimator(graspologic.embed.AdjacencySpectralEmbed)  # TODO: figure out a better solution here
=======
# TODO: figure out a better solution here
# check_estimator(graspologic.embed.AdjacencySpectralEmbed)  # with current implementation of predict, this class is no longer sklearn compliant
>>>>>>> 58c2d46107e22c2156f1649e7f7127251c6c91c2
check_estimator(graspologic.embed.LaplacianSpectralEmbed)
check_estimator(graspologic.embed.ClassicalMDS)
