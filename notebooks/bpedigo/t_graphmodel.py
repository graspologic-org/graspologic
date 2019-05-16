#%%
import graspy
from graspy.models.sbm import SBEstimator
from graspy.simulations import sbm
import numpy as np
from graspy.plot import heatmap


n_verts = 1000
n = 2 * [n_verts]
# print(n)
b = np.array([[0.8, 0.4], [0.1, 0.5]])
labels = n_verts * ["1"] + n_verts * [2]
graph = sbm(n, b, directed=True)
sb = SBEstimator(fit_degrees=False)
b = sb.fit(graph, labels).block_p_

heatmap(graph)
heatmap(sb.sample())
# import itertools
# from collections import Counter

# def other(graph, labels):
#     u, inv, counts = np.unique(labels, return_inverse=True, return_counts=True)
#     block_probs = np.zeros((len(u), len(u)))
#     block_sizes = np.zeros((len(u), len(u)))
#     nonzero_inds = np.nonzero(graph)
#     from_inds = inv[nonzero_inds[0]]
#     to_inds = inv[nonzero_inds[1]]

#     size_product = itertools.product(counts, counts)


#     c = Counter(elem for elem in zip(from_inds, to_inds))
#     for counts, sizes in zip(c.items(), size_product):
#         block_probs[counts[0]] = counts[1]
#         block_sizes[counts[0]] = sizes[0] * sizes[1]
#     block_probs /= block_sizes
#     block_probs

# other(graph, labels)

#%%
from graspy.models.er import SBEstimator
from graspy.simulations import er_np

graph = er_np(100, 0.24)
er = SBEstimator()
er.fit(graph)
er.p_

#%%


#%%
