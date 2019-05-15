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
thetas = np.array([0.0 * np.pi, 0.45 * np.pi])
distances = np.random.beta(1.5, 2, n_verts)
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
