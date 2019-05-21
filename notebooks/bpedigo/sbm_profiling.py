#%%
import matplotlib.pyplot as plt
import seaborn as sns
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.models import DCSBEstimator, SBEstimator, BaseGraphEstimator
from graspy.plot import heatmap
from graspy.simulations import p_from_latent, sample_edges, sbm
from graspy.utils import *


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


n_verts = 200
show_graphs = False
show_latent = True
p_kwargs = {}
sample_kwargs = {}

# dcsbm, 2 line, beta
thetas = np.array([0.0 * np.pi, 0.42 * np.pi])
distances = np.random.beta(4, 1, n_verts)
vec1 = np.array([np.cos(thetas[0]), np.sin(thetas[0])])
vec2 = np.array([np.cos(thetas[1]), np.sin(thetas[1])])
latent1 = np.multiply(distances[: int(n_verts / 2)][:, np.newaxis], vec1[np.newaxis, :])
latent2 = np.multiply(distances[int(n_verts / 2) :][:, np.newaxis], vec2[np.newaxis, :])
latent = np.concatenate((latent1, latent2), axis=0)
labels = np.array(latent.shape[0] // 2 * ["0"] + latent.shape[0] // 2 * ["1"])
dcsbm_P = p_from_latent(latent, rescale=False, loops=False)
graph = sample_edges(dcsbm_P, directed=False, loops=False)
# graph = get_graph(latent, "DCSBM", labels=labels)


graph
heatmap(graph, inner_hier_labels=labels)

dcsbe = DCSBEstimator(directed=False, loops=False)
dcsbe.fit(graph)
dcsbe.degree_corrections_
plt.figure()
sns.distplot(dcsbe.degree_corrections_)
plt.figure()
sns.distplot(distances)
p_hat = dcsbe.p_mat_

np.linalg.norm(p_hat - dcsbm_P) ** 2


plt.figure()
sns.scatterplot(
    x=latent[:, 0], y=latent[:, 1], hue=dcsbe.vertex_assignments_, linewidth=0
)


# plt.style.use("seaborn-white")
# sns.set_palette("Set1")
plt.figure(figsize=(10, 10))
sns.set_context("talk", font_scale=1.5)
sns.scatterplot(x=latent[:, 0], y=latent[:, 1], hue=labels, linewidth=0)
plt.axis("square")
ase = AdjacencySpectralEmbed(n_components=2)
lse = LaplacianSpectralEmbed(n_components=2, form="R-DAD", regularizer=1)
ase_latent = ase.fit_transform(graph)
lse_latent = lse.fit_transform(graph)

plt.figure(figsize=(10, 10))
sns.scatterplot(x=ase_latent[:, 0], y=ase_latent[:, 1], hue=labels, linewidth=0)
plt.axis("square")

plt.figure(figsize=(10, 10))
sns.scatterplot(x=lse_latent[:, 0], y=lse_latent[:, 1], hue=labels, linewidth=0)
plt.axis("square")
vector_lengths = np.linalg.norm(ase_latent, axis=1)
vector_lengths[vector_lengths == 0] = 1
proj_latent = ase_latent / vector_lengths[:, np.newaxis]

plt.figure(figsize=(10, 10))
sns.scatterplot(x=proj_latent[:, 0], y=proj_latent[:, 1], hue=labels, linewidth=0)
plt.axis("square")


#%% Experiment: with known block assignments, how well can we recover DCSBM params?
n_verts = 1000
# dcsbm, 2 line, beta
thetas = np.array([0.0 * np.pi, 0.4 * np.pi])
distances = np.random.beta(1.5, 2, n_verts)
vec1 = np.array([np.cos(thetas[0]), np.sin(thetas[0])])
vec2 = np.array([np.cos(thetas[1]), np.sin(thetas[1])])
latent1 = np.multiply(distances[: int(n_verts / 2)][:, np.newaxis], vec1[np.newaxis, :])
latent2 = np.multiply(distances[int(n_verts / 2) :][:, np.newaxis], vec2[np.newaxis, :])
latent = np.concatenate((latent1, latent2), axis=0)

# thetas = np.array([0 * np.pi, 0.4 * np.pi])
# distances = np.random.beta(1.5, 2, n_verts)
# vec1 = np.array([np.cos(thetas[0]), np.sin(thetas[0])])
# vec2 = np.array([np.cos(thetas[1]), np.sin(thetas[1])])
# latent1 = np.multiply(distances[: int(n_verts / 2)][:, np.newaxis], vec1[np.newaxis, :])
# latent2 = np.multiply(distances[int(n_verts / 2) :][:, np.newaxis], vec2[np.newaxis, :])
# latent_right = np.concatenate((latent1, latent2), axis=0)

labels = np.array(latent.shape[0] // 2 * ["0"] + latent.shape[0] // 2 * ["1"])
dcsbm_P = p_from_latent(latent, latent, rescale=False, loops=False)
graph = sample_edges(dcsbm_P, directed=False, loops=False)
heatmap(graph, inner_hier_labels=labels)

# Plot OG latent positions
plt.figure(figsize=(10, 10))
sns.set_context("talk", font_scale=1.5)
sns.scatterplot(x=latent[:, 0], y=latent[:, 1], hue=labels, linewidth=0)
plt.axis("square")
plt.title("True latent positions")

dcsbe = DCSBEstimator(directed=False, loops=False)
dcsbe.fit(graph, y=labels)
dcsbe.degree_corrections_
plt.figure()
sns.distplot(distances)
plt.title("True distances")
plt.figure()
sns.distplot(dcsbe.degree_corrections_)
plt.title("Estimated distance distribution")

heatmap(dcsbm_P, inner_hier_labels=labels)
heatmap(dcsbe.p_mat_, inner_hier_labels=labels)
#%%
# Figure this out
sbe = SBEstimator()

b = dcsbm_P / np.outer(distances, distances)


c = dcsbm_P / np.outer(d, d)
heatmap(b, inner_hier_labels=labels)
heatmap(c)
#%% regular sbm
from graspy.simulations import sbm
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from graspy.embed import AdjacencySpectralEmbed, LaplacianSpectralEmbed
from graspy.models import DCSBEstimator, SBEstimator
from graspy.plot import heatmap
from graspy.simulations import p_from_latent, sample_edges, sbm
from graspy.utils import *


B = np.array(
    [
        [0.9, 0.2, 0.05, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.2, 0.4, 0.8, 0.5],
        [0.1, 0.2, 0.1, 0.7],
    ]
)
n = [100, 20, 30, 50]
labels = np.array(
    list(itertools.chain.from_iterable([m * [i] for i, m in enumerate(n)]))
)

g = sbm(n, B, directed=True, loops=False)
sbe = SBEstimator(directed=True)
sbe.fit(g, labels)
sbe.block_p_
sbe._n_parameters()
g = sbe.sample(10)
heatmap(g[0])


#%%
n = np.array([100, 20, 30, 50])


def _n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum())
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


#%%
from graspy.utils import cartprod


def _n_to_labels(n):
    n_cumsum = n.cumsum()
    labels = np.zeros(n.sum(), dtype=np.int64)
    for i in range(1, len(n)):
        labels[n_cumsum[i - 1] : n_cumsum[i]] = i
    return labels


def _block_to_full(block_mat, inverse, shape):
    """
    "blows up" a k x k matrix, where k is the number of communities, 
    into a full n x n probability matrix

    block mat : k x k 
    inverse : array like length n, 
    """
    block_map = cartprod(inverse, inverse).T
    mat_by_edge = block_mat[block_map[0], block_map[1]]
    full_mat = mat_by_edge.reshape(shape)
    return full_mat


def cartprod(*arrays):
    N = len(arrays)
    return np.transpose(
        np.meshgrid(*arrays, indexing="ij"), np.roll(np.arange(N + 1), -1)
    ).reshape(-1, N)


np.random.seed(8888)
B = np.array(
    [
        [0.9, 0.2, 0.05, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.2, 0.4, 0.8, 0.5],
        [0.1, 0.2, 0.1, 0.7],
    ]
)
n = np.array([1000, 1000, 500, 500])
# g = sbm(n, B, directed=True, loops=False)
dc = np.random.beta(2, 5, size=n.sum())
labels = _n_to_labels(n)
p_mat = _block_to_full(B, labels, (n.sum(), n.sum()))
p_mat = p_mat * np.outer(dc, dc)
p_mat -= np.diag(np.diag(p_mat))
g = sample_edges(p_mat, directed=True, loops=False)
dcsbe = DCSBEstimator(directed=True, loops=False)
dcsbe.fit(g, y=labels)
diff = dcsbe.p_mat_ - p_mat
heatmap(p_mat, inner_hier_labels=labels)
heatmap(dcsbe.p_mat_, inner_hier_labels=labels)
#%%
np.allclose(dcsbe.p_mat_, p_mat, atol=0.1)


#%%

#%%
from sklearn.metrics import adjusted_rand_score
from sklearn.manifold import MDS
from graspy.embed import ClassicalMDS
from graspy.plot import pairplot
from graspy.cluster import GaussianCluster


# thetas = np.array([0.0 * np.pi, 0.42 * np.pi])
# vec1 = np.array([np.cos(thetas[0]), np.sin(thetas[0])])
# vec2 = np.array([np.cos(thetas[1]), np.sin(thetas[1])])
# latent1 = np.multiply(distances[: int(n_verts / 2)][:, np.newaxis], vec1[np.newaxis, :])
# latent2 = np.multiply(distances[int(n_verts / 2) :][:, np.newaxis], vec2[np.newaxis, :])
# latent = np.concatenate((latent1, latent2), axis=0)
# dcsbm_P = p_from_latent(latent, rescale=False, loops=False)

graph = sample_edges(p_mat, directed=True, loops=False)
latent = AdjacencySpectralEmbed(algorithm="full", n_components=3).fit_transform(graph)
latent = np.concatenate(latent, axis=1)
pairplot(latent)
dcsbe = DCSBEstimator(directed=True, loops=False)
dcsbe.fit(graph)
adjusted_rand_score(labels, dcsbe.vertex_assignments_) > 0.95

plt.figure(figsize=(10, 10))
sns.set_context("talk", font_scale=1.5)
# sns.scatterplot(x=latent[:, 0], y=latent[:, 1], hue=labels, linewidth=0)
pairplot(latent, labels=labels)
plt.axis("square")
plt.title("ASE - True labels")

ase = AdjacencySpectralEmbed(n_components=3)
lse = LaplacianSpectralEmbed(n_components=3, form="R-DAD", regularizer=1)
ase_latent = ase.fit_transform(graph)
ase_latent = np.concatenate(ase_latent, axis=1)
lse_latent = lse.fit_transform(symmetrize(graph, "avg"))

plt.figure(figsize=(10, 10))
# sns.scatterplot(
#     x=ase_latent[:, 0], y=ase_latent[:, 1], hue=dcsbe.vertex_assignments_, linewidth=0
# )
pairplot(lse_latent, labels=dcsbe.vertex_assignments_)
plt.axis("square")
plt.title("ASE - pred labels GMM")

plt.figure(figsize=(10, 10))
sns.scatterplot(x=lse_latent[:, 0], y=lse_latent[:, 1], hue=labels, linewidth=0)
plt.axis("square")
plt.title("LSE - true labels")
gc = GaussianCluster(max_components=5, covariance_type="all")
lse_labels = gc.fit_predict(lse_latent)
plt.figure(figsize=(10, 10))
# sns.scatterplot(x=lse_latent[:, 0], y=lse_latent[:, 1], hue=lse_labels, linewidth=0)
#%%
pairplot(lse_latent, labels=lse_labels)
plt.axis("square")
plt.title("LSE - pred labels")

vector_lengths = np.linalg.norm(ase_latent, axis=1)
vector_lengths[vector_lengths == 0] = 1
proj_latent = ase_latent / vector_lengths[:, np.newaxis]
# plt.figure(figsize=(10, 10))
# sns.scatterplot(x=proj_latent[:, 0], y=proj_latent[:, 1], hue=labels, linewidth=0)
# plt.axis("square")
# assert_allclose(dcsbm_P, dcsbe.p_mat_, atol=0.1)

#%%
proj_dot = proj_latent @ proj_latent.T
proj_sphere_dist = np.nan_to_num(np.arccos(proj_dot))
proj_sphere_dist -= np.diag(np.diag(proj_sphere_dist))
# proj_sphere_dist += np.diag(np.ones(proj_sphere_dist.shape[0]))
heatmap(proj_sphere_dist, inner_hier_labels=labels)

#%%
cmds = ClassicalMDS(dissimilarity="precomputed", n_components=proj_latent.shape[1] - 1)
embed_sphere_dist = cmds.fit_transform(proj_sphere_dist)
# sns.distplot(embed_sphere_dist)
pairplot(embed_sphere_dist, labels=labels)

mds = MDS(
    dissimilarity="precomputed", n_components=proj_latent.shape[1] - 1, metric=True
)
# embed_sphere_dist = mds.fit_transform(proj_sphere_dist)

pairplot(embed_sphere_dist, labels=labels)
gc = GaussianCluster(max_components=6)
sphere_pred_labels = gc.fit_predict(embed_sphere_dist)

adjusted_rand_score(labels, sphere_pred_labels)
pairplot(embed_sphere_dist, labels=sphere_pred_labels)
# adjusted_rand_score(labels, dcsbe.vertex_assignments_)


#%%
np.random.seed(1234)
n_verts = 750

B = np.array([[0.7, 0.1, 0.1], [0.1, 0.9, 0.1], [0.05, 0.1, 0.75]])
n = np.array([250, 250, 250])
labels = _n_to_labels(n)
p_mat = _block_to_full(B, labels, (n_verts, n_verts))
p_mat = p_mat * np.outer(distances, distances)

graph = sample_edges(p_mat, directed=True, loops=False)
dcsbe = DCSBEstimator(directed=True, loops=False)
dcsbe.fit(graph)
adjusted_rand_score(labels, dcsbe.vertex_assignments_)


#%%
np.random.seed(8888)
B = np.array(
    [
        [0.9, 0.2, 0.05, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.2, 0.4, 0.8, 0.5],
        [0.1, 0.2, 0.1, 0.7],
    ]
)
n = np.array([1000, 1000, 500, 500])
dc = np.random.beta(2, 5, size=n.sum())
labels = _n_to_labels(n)
p_mat = _block_to_full(B, labels, (n.sum(), n.sum()))
p_mat = p_mat * np.outer(dc, dc)
p_mat -= np.diag(np.diag(p_mat))
g = sample_edges(p_mat, directed=True, loops=False)
dcsbe = DCSBEstimator()
dcsbe.fit(g)
heatmap(dcsbe.p_mat_)
