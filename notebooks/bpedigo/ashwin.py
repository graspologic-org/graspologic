#%%
from scipy.io import loadmat

adj = loadmat("/Users/bpedigo/JHU_code/graspy/notebooks/bpedigo/ConnMatrixPre.mat")[
    "ConnMatrixPre"
]
from graspy.plot import heatmap, gridplot, pairplot

adj
heatmap(adj, transform="simple-nonzero")

gridplot([adj], labels=[1])
# presynaptic on the columns

from graspy.embed import AdjacencySpectralEmbed
from graspy.utils import pass_to_ranks, get_lcc, augment_diagonal
from graspy.cluster import GaussianCluster


ase = AdjacencySpectralEmbed()
adj_lcc = get_lcc(adj)
adj_ptr = pass_to_ranks(adj_lcc)
adj_ptr = augment_diagonal(adj_ptr)
X, Y = ase.fit_transform(adj_ptr)

latent = np.concatenate((X, Y), axis=1)
pairplot(latent)
gc = GaussianCluster(max_components=10)
gc_labels = gc.fit_predict(latent)

pairplot(latent[:, :2], labels=gc_labels, height=5)


gridplot(
    [adj_ptr],
    labels=[1],
    inner_hier_labels=gc_labels,
    height=20,
    alpha=0.3,
    sizes=(5, 50),
)

#%%
from graspy.plot import screeplot

screeplot(adj_ptr)
#%%
plt.plot(gc.bic_)
#%%

#%%
plt.figure(figsize=(20, 20))
w, v = np.linalg.eig(adj_ptr)
plt.plot(w)
