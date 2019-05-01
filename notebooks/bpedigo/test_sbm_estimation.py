#%%
from graspy.embed import AdjacencySpectralEmbed
from graspy.models import SBEstimator
from graspy.simulations import p_from_latent, sample_edges, sbm
from graspy.utils import *


def spectral_fit_sbm(graph, memberships):
    ase = AdjacencySpectralEmbed(n_components=4)
    graph = augment_diagonal(graph)
    X, Y = ase.fit_transform(graph)
    latent = np.concatenate((X, Y), axis=1)
    blocks = np.unique(memberships)
    block_centroids = []
    for b in blocks:
        inds = np.where(memberships == b)[0]
        centroid = np.mean(latent[inds, :], axis=0)
        block_centroids.append(centroid)
    block_centroids = np.array(block_centroids)
    return block_centroids @ block_centroids.T


def get_graph(latent, title=None, labels=None):
    if type(latent) is tuple:
        left_latent = latent[0]
        right_latent = latent[1]
    else:
        left_latent = latent
        right_latent = None
    true_P = p_from_latent(left_latent, right_latent, **p_kwargs)
    graph = sample_edges(true_P, **sample_kwargs)

    return graph


B = np.array(
    [
        [0.9, 0.2, 0.05, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.2, 0.4, 0.8, 0.5],
        [0.1, 0.2, 0.1, 0.7],
    ]
)

# B = np.full((4, 4), 0.3)
n_total = 1000
block_counts = n_total * np.array([0.2, 0.5, 0.2, 0.1])
block_counts = block_counts.astype(int)
labels = np.zeros(n_total, dtype=int)
count = 0
cluster_kws = {"max_components": 10}
for i, c in enumerate(block_counts):
    for j in range(c):
        labels[j + count] = i
    count = count + c


simple_error = []
spectral_error = []
n_sims = 1
for i in range(n_sims):
    sample = sbm(block_counts, B, directed=True, loops=True)
    sbe = SBEstimator(cluster_kws=cluster_kws)
    sbe.fit(sample)
    B_hat_simple = sbe.block_p_

    B_hat_spectral = spectral_fit_sbm(sample, labels)
    simple_error.append(np.mean((B - B_hat_simple) ** 2))
    spectral_error.append(np.mean((B - B_hat_spectral) ** 2))

print(np.sum(simple_error))
print(np.sum(spectral_error))

#%%
n_verts = 200
show_graphs = False
show_latent = True
p_kwargs = {}
sample_kwargs = {}

# dcsbm, 2 line, beta
thetas = np.array([0.0 * np.pi, 0.4 * np.pi])
distances = np.random.beta(1, 0.5, n_verts)
vec1 = np.array([np.cos(thetas[0]), np.sin(thetas[0])])
vec2 = np.array([np.cos(thetas[1]), np.sin(thetas[1])])
latent1 = np.multiply(distances[: int(n_verts / 2)][:, np.newaxis], vec1[np.newaxis, :])
latent2 = np.multiply(distances[int(n_verts / 2) :][:, np.newaxis], vec2[np.newaxis, :])
latent = np.concatenate((latent1, latent2), axis=0)
labels = np.array(latent.shape[0] // 2 * ["0"] + latent.shape[0] // 2 * ["1"])
dcsbm_P = p_from_latent(latent, rescale=False, loops=False)
graph = sample_edges(dcsbm_P, directed=False, loops=False)
# graph = get_graph(latent, "DCSBM", labels=labels)

#%%
from graspy.plot import heatmap

graph
heatmap(graph, inner_hier_labels=labels)

dcsbe = SBEstimator(fit_degrees=True, directed=False, loops=False)
dcsbe.fit(graph)
dcsbe.degree_corrections_
plt.figure()
sns.distplot(dcsbe.degree_corrections_)
sns.distplot(distances)
p_hat = dcsbe.p_mat_

np.linalg.norm(p_hat - dcsbm_P) ** 2

#%%
