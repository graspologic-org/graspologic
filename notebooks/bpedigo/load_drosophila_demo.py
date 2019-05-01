#%%
from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.plot import heatmap, pairplot
from graspy.utils import symmetrize, binarize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

left_adj, left_labels = load_drosophila_left(return_labels=True)
right_adj, right_labels = load_drosophila_right(return_labels=True)

heatmap(left_adj, inner_hier_labels=left_labels)
heatmap(right_adj, inner_hier_labels=right_labels)

heatmap(left_adj, transform="simple-nonzero", inner_hier_labels=left_labels)
heatmap(right_adj, transform="simple-nonzero", inner_hier_labels=right_labels)


heatmap(left_adj, transform="zero-boost", inner_hier_labels=left_labels)
heatmap(right_adj, transform="zero-boost", inner_hier_labels=right_labels)

ase = AdjacencySpectralEmbed()
right_adj = binarize(right_adj)
latent = ase.fit_transform(right_adj)
latent = np.concatenate(latent, axis=1)
pairplot(latent, labels=right_labels, diag_kind="hist")
