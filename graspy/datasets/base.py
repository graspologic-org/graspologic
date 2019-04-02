from os.path import dirname, join
import numpy as np


def load_drosophila_left(return_labels=False):
    module_path = dirname(__file__)
    folder = "drosophila"
    filename = "left_adjacency.csv"
    with open(join(module_path, folder, filename)) as csv_file:
        graph = np.loadtxt(csv_file, dtype=int)
    if return_labels:
        filename = "left_cell_labels.csv"
        with open(join(module_path, folder, filename)) as csv_file:
            labels = np.loadtxt(csv_file, dtype=str)
        return graph, labels
    else:
        return graph
