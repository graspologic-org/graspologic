#%%
from graspy.embed import AdjacencySpectralEmbed
from graspy.utils import *
from graspy.models import SBEstimator
from graspy.simulations import sbm

B = np.array(
    [
        [0.9, 0.2, 0.05, 0.1],
        [0.1, 0.7, 0.1, 0.1],
        [0.2, 0.4, 0.8, 0.5],
        [0.1, 0.2, 0.1, 0.7],
    ]
)

B = np.full((4, 4), 0.3)
n_total = 1000
block_counts = n_total * np.array([0.2, 0.5, 0.2, 0.1])
block_counts = block_counts.astype(int)
labels = np.zeros(n_total, dtype=int)
count = 0
for i, c in enumerate(block_counts):
    for j in range(c):
        labels[j + count] = i
    count = count + c


simple_error = []
spectral_error = []
n_sims = 10
for i in range(n_sims):
    sample = sbm(block_counts, B, directed=True, loops=True)
    sbe = SBEstimator()
    sbe.fit(sample, labels)
    B_hat_simple = sbe.block_p_

    def spectral_fit_sbm(graph, memberships):
        ase = AdjacencySpectralEmbed(n_components=4)
        graph = augment_diagonal(graph)
        X, Y = ase.fit_transform(graph)
        print(X.shape)
        latent = np.concatenate((X, Y), axis=1)
        blocks = np.unique(memberships)
        block_centroids = []
        for b in blocks:
            inds = np.where(memberships == b)[0]
            centroid = np.mean(latent[inds, :], axis=0)
            block_centroids.append(centroid)
            print(centroid.shape)
        block_centroids = np.array(block_centroids)
        print(block_centroids.shape)
        return block_centroids @ block_centroids.T

    B_hat_spectral = spectral_fit_sbm(sample, labels)
    simple_error.append(np.mean((B - B_hat_simple) ** 2))
    spectral_error.append(np.mean((B - B_hat_spectral) ** 2))

print(np.sum(simple_error))
print(np.sum(spectral_error))

#%%


#%%
