#%%
# %matplotlib inline
from graspy.plot import *
from graspy.simulations import sbm
from graspy.embed import AdjacencySpectralEmbed
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

B = np.array(
    [
        [0, 0.2, 0.1, 0.1, 0.1],
        [0.2, 0.8, 0.1, 0.3, 0.1],
        [0.15, 0.1, 0, 0.05, 0.1],
        [0.1, 0.1, 0.2, 1, 0.1],
        [0.1, 0.2, 0.1, 0.1, 0.8],
    ]
)

g = sbm([10, 30, 50, 25, 25], B, directed=True)
ase = AdjacencySpectralEmbed()
X = ase.fit_transform(g)
labels2 = 40 * ["0"] + 100 * ["1"]
# pairplot(X, size=50, alpha=0.6)
labels1 = 10 * ["d"] + 30 * ["c"] + 50 * ["d"] + 25 * ["e"] + 25 * ["c"]
labels1 = np.array(labels1)
labels2 = np.array(labels2)
heatmap(g, inner_hier_labels=labels1, outer_hier_labels=labels2, figsize=(30, 30))


def _get_freq_maps(inner_labels, outer_labels):
    outer_unique, outer_inv, outer_freq = np.unique(
        outer_labels, return_counts=True, return_inverse=True
    )
    outer_freq_cumsum = np.hstack((0, outer_freq.cumsum()))
    outer_label_counts = outer_freq[outer_inv]

    inner_label_counts = np.zeros_like(outer_label_counts)
    for i, outer_label in enumerate(outer_unique):
        outer_label_inds = np.where(outer_inv == i)[0]
        print(outer_label_inds)
        temp_inner_labels = inner_labels[outer_label_inds]
        temp_inner_unique, temp_inv, temp_freq = np.unique(
            temp_inner_labels, return_inverse=True, return_counts=True
        )
        temp_inner_label_counts = temp_freq[temp_inv]
        inner_label_counts[outer_label_inds] = temp_inner_label_counts

    return inner_label_counts, outer_label_counts
    #         inner_labels[start_ind:stop_ind], return_counts=True
    #     )
    # for each group of outer labels, calculate the boundaries of the inner labels
    # inner_freq = np.array([])
    # inner_unique = np.array([])
    # inner_label_counts = np.zeros(inner_labels.shape[0])
    # for i in range(outer_freq.size):
    #     start_ind = outer_freq_cumsum[i]
    #     stop_ind = outer_freq_cumsum[i + 1]
    #     temp_inner_unique, temp_freq = np.unique(
    #         inner_labels[start_ind:stop_ind], return_counts=True
    #     )

    #     inner_freq = np.hstack([inner_freq, temp_freq])
    #     inner_unique = np.hstack([inner_unique, temp_inner_unique])
    #     temp_inner_map = dict(zip(inner_unique, inner_freq))
    #     temp_inner_mapped = itemgetter(*inner_labels[start_ind:stop_ind])(
    #         temp_inner_map
    #     )
    #     inner_label_counts[start_ind:stop_ind] = temp_inner_mapped

    # print(inner_label_counts)
    # outer_map = dict(zip(outer_unique, outer_freq))
    # outer_label_counts = itemgetter(*outer_labels)(outer_map)

    # return inner_label_counts, outer_label_counts


plt.show()
#%%
_get_freq_maps(labels1, labels2)
