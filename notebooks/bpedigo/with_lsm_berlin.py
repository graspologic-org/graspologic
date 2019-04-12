#%%
from graspy.utils import *
from graspy.plot import *
from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.simulations import sample_edges, p_from_latent
import numpy as np
import pandas as pd

means = pd.read_csv("~/JHU_code/mbstructure/means.csv").values
mean1 = pd.read_csv("~/JHU_code/mbstructure/mean1.csv").values.T
mean2 = pd.read_csv("~/JHU_code/mbstructure/mean2.csv").values.T
mean3 = pd.read_csv("~/JHU_code/mbstructure/mean3.csv").values.T

xhat = pd.read_csv("~/JHU_code/mbstructure/Xhat.csv").values
pairplot(means)
right_adj_raw, right_labels = load_drosophila_right(return_labels=True)

#%%
struct_latent = means

for l, m in zip(["I", "O", "P"], [mean1, mean2, mean3]):
    n_cells = (right_labels == l).sum()
    latent_mat = np.zeros((n_cells, 6))
    latent_mat[:, :] = m[np.newaxis, :]
    struct_latent = np.concatenate((struct_latent, latent_mat), axis=0)

#%%
pairplot(struct_latent)
pairplot(xhat)
pairplot(np.concatenate((xhat, means), axis=0))
struct_latent = means
for l in ["I", "O", "P"]:
    mean = np.mean(xhat[right_labels == l], axis=0)
    print(mean)
    n_cells = (right_labels == l).sum()
    latent_mat = np.zeros((n_cells, 6))
    for i, c in enumerate(xhat[right_labels == l]):
        proj = np.dot(c, mean)
        proj_vec = proj * mean
        latent_mat[i, :] = proj_vec

    # latent_mat[:, :] = mean[np.newaxis, :]
    struct_latent = np.concatenate((struct_latent, latent_mat), axis=0)
pairplot(struct_latent)

p_mat = p_from_latent(
    struct_latent[:, :3], struct_latent[:, 3:], rescale=False, loops=False
)
graph = sample_edges(p_mat, directed=True, loops=False)
#%%
heatmap(graph, inner_hier_labels=right_labels)
heatmap(right_adj_raw, inner_hier_labels=right_labels, transform="zero-boost")

