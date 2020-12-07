from .base import BaseSpectralEmbed
import numpy as np


class CASC(BaseSpectralEmbed):
    # TODO: everything
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.is_fitted_ = False

    def fit(self, graph, y=None):
        self.is_fitted_ = True
        pass

    def transform(self, graph, y=None):
        A = import_graph(graph)
        pass


def gen_covariates(m1, m2, labels):
    # TODO: make sure labels is 1d array-like
    n = len(labels)
    m1_arr = np.random.choice([1, 0], p=[m1, 1 - m1], size=(n))
    m2_arr = np.random.choice([1, 0], p=[m2, 1 - m2], size=(n, 3))
    m2_arr[np.arange(n), labels] = m1_arr
    return m2_arr