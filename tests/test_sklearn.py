import graspy
import numpy as np

from sklearn.utils.estimator_checks import check_estimator

check_estimator(graspy.embed.AdjacencySpectralEmbed)
# check_estimator(graspy.embed.AdjacencySpectralEmbed(algorithm="full"))
# check_estimator(graspy.embed.AdjacencySpectralEmbed(algorithm="truncated"))
# check_estimator(graspy.embed.LaplacianSpectralEmbed)
