#%%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
import seaborn as sns
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import *
from sklearn.metrics import adjusted_rand_score

# sim_mat = augment_diagonal(sim_mat)
from sklearn.utils.graph_shortest_path import graph_shortest_path

from graspy.cluster import GaussianCluster
from graspy.datasets import load_drosophila_left, load_drosophila_right

#%%
from graspy.embed import AdjacencySpectralEmbed, ClassicalMDS, LaplacianSpectralEmbed
from graspy.plot import heatmap, pairplot
from graspy.utils import augment_diagonal, binarize, is_fully_connected, symmetrize

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
#%%
right_adj = binarize(right_adj)
plt.figure(figsize=(10, 10))
w, v = np.linalg.eig(right_adj)
w = np.abs(w)
w = np.sort(w)[::-1]
plt.plot(w, ".")


AdjacencySpectralEmbed()


is_fully_connected(right_adj)

simfile = "/Users/bpedigo/JHU_code/graspy/notebooks/similarityMatrix.csv"
simfile = "/Users/bpedigo/JHU_code/graspy/notebooks/bpedigo/drosophila.csv"

sim_mat = pd.read_csv(simfile, header=None)
sim_mat = sim_mat.values

G = graph_shortest_path(1 - sim_mat)

# sim_mat=G

heatmap(sim_mat, inner_hier_labels=right_labels)
# sim_mat = augment_diagonal(sim_mat)
# sim_mat = sim_mat + 2000
ase = AdjacencySpectralEmbed(n_components=5)
latent = ase.fit_transform(sim_mat)
pairplot(latent, labels=right_labels)

#%%
mds = MDS(n_components=4, metric=False, dissimilarity="precomputed")
mds_l = mds.fit_transform(1 - sim_mat)
pairplot(mds_l, labels=right_labels)
#%%
se = SpectralEmbedding(n_components=4, affinity="precomputed")
se_embed = se.fit_transform(sim_mat)
pairplot(se_embed, labels=right_labels)
#%%
se_embed2 = spectral_embedding(sim_mat, n_components=4, norm_laplacian=True)
pairplot(se_embed2, labels=right_labels)

tsne_embed = TSNE(n_components=3, metric="precomputed").fit_transform(sim_mat)
pairplot(tsne_embed, labels=right_labels)


#%%
phate_op = phate.PHATE(n_components=4, knn_dist="precomputed").fit_transform(
    1 - sim_mat
)
pairplot(phate_op, labels=right_labels)


cmds_embed = ClassicalMDS(n_components=4).fit_transform(sim_mat)
pairplot(cmds_embed, labels=right_labels)

lse = LaplacianSpectralEmbed(n_components=4, form="R-DAD", regularizer=10)
latent_lse = lse.fit_transform(sim_mat)
pairplot(latent_lse, labels=right_labels)

ase = AdjacencySpectralEmbed(n_components=4)
latent = ase.fit_transform(sim_mat)
pairplot(latent, labels=right_labels)


#%%

#%%


ag = AgglomerativeClustering(n_clusters=5, affinity="precomputed", linkage="average")
ag_predict = ag.fit_predict(1 - sim_mat)
# heatmap(c/edict, labels=right_labels)
adjusted_rand_score(right_labels, ag_predict)

heatmap(right_adj, inner_hier_labels=ag_predict)


#%%
heatmap(right_adj)

#%%

#%%


embedding = umap.UMAP(metric="precomputed", n_components=4).fit_transform(1 - sim_mat)

pairplot(embedding, labels=right_labels)


gc = GaussianCluster(min_components=1, max_components=15, covariance_type="all")
gc_predict = gc.fit_predict(latent)

adjusted_rand_score(right_labels, gc_predict)


#%%

#%%
from sklearn.cluster import DBSCAN

db = DBSCAN(metric="precomputed", eps=0.5, min_samples=5)
db_predict = db.fit_predict(1 - sim_mat)
adjusted_rand_score(right_labels, db_predict)

pairplot(latent, labels=db_predict)

#%%

from graspy.plot import edgeplot

edgeplot(sim_mat)
