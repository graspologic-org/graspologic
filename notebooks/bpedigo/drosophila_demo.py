#%%

from graspy.models import EREstimator, SBEstimator, RDPGEstimator
from graspy.datasets import load_drosophila_left
from graspy.plot import heatmap
import numpy as np

left_adj, cell_labels = load_drosophila_left(return_labels=True)
left_adj_uw = left_adj.copy()
left_adj_uw[left_adj_uw > 0] = 1

# heatmap(left_adj_uw, inner_hier_labels=cell_labels)

models = [
    EREstimator(fit_degrees=False),
    EREstimator(fit_degrees=True),
    SBEstimator(fit_degrees=False),
    SBEstimator(fit_degrees=True),
    RDPGEstimator(n_components=50),
    RDPGEstimator(n_components=10),
]

for model in models:
    m = model.fit(left_adj, y=cell_labels)
    # heatmap(m.p_mat_, inner_hier_labels=cell_labels)
    # heatmap(m.sample(), inner_hier_labels=cell_labels)
    s = m.sample()
    # print(m.score(s))
    # print(m.score(left_adj))
    # heatmap(m.score_samples(left_adj))
    # print(m.bic(left_adj))
    # print(m._n_parameters())
    edge_log_likelihoods = m.score_samples(left_adj)
    inds = np.where(np.logical_and(np.isneginf(edge_log_likelihoods), left_adj == 0))
    print(len(inds[0]))
    print(np.isneginf(edge_log_likelihoods).sum())
    edge_log_likelihoods[np.isneginf(edge_log_likelihoods)] = 0
    print(edge_log_likelihoods.sum())
    p_mat = m.p_mat_
    print(p_mat.max())
    print(p_mat.min())
    print(p_mat[p_mat == 1.0].sum())
    print()
    print()
#%%
# er = EREstimator(fit_degrees=False)
# er.fit(left_adj)
# heatmap(er.sample(), inner_hier_labels=cell_labels)

# dcer = EREstimator(fit_degrees=True)
# dcer.fit(left_adj)
# heatmap(dcer.sample(), inner_hier_labels=cell_labels)

# sbm = SBEstimator(fit_degrees=False)
# sbm.fit(left_adj, y=cell_labels)
# heatmap(sbm.sample(), inner_hier_labels=cell_labels)

# dcsbm = SBEstimator(fit_degrees=True)
# dcsbm.fit(left_adj, y=cell_labels)
# d = dcsbm.sample()
# heatmap(d, inner_hier_labels=cell_labels)


# sbm_test = SBEstimator(fit_degrees=False)
# sbm_test.fit(d, y=cell_labels)
# sbm_test.block_p_
# sbm.block_p_
# heatmap(dcsbm.p_mat_)
#%%
from graspy.simulations import er_np

n_sims = 10
ll_m = np.zeros(n_sims)
ll_e = []
p = 0.2
for i in range(n_sims):
    g = er_np(100, p, directed=True, loops=True)
    er = EREstimator(False)
    er = er.fit(g)
    log_likelihood_edges = er.score_samples(g)
    log_likelihood_mean = er.score(g)
    # er.bic(g)
    ll_m[i] = log_likelihood_mean
    ll_e.append(log_likelihood_edges)

import seaborn as sns

likelihood = np.exp(er.score_samples(g))
likelihood.max()
likelihood.min()
heatmap(likelihood)
heatmap(g)
print(p * p + (1 - p) * (1 - p))
print(ll_m.mean())
# print(np.exp(ll_m.mean()))

#%%
def sbm_p_matrix(block_n, block_ps, dc=None):
    block_members = []
    for i, bs in enumerate(block_n):
        block_members = block_members + bs * [i]
    block_members = np.array(block_members)
    block_map = cartprod(block_members, block_members).T
    p_by_edge = block_ps[block_map[0], block_map[1]]
    p_mat = p_by_edge.reshape((m.sum(), m.sum()))
    return p_mat


def cartprod(*arrays):
    N = len(arrays)
    return np.transpose(
        np.meshgrid(*arrays, indexing="ij"), np.roll(np.arange(N + 1), -1)
    ).reshape(-1, N)


m = np.array([10, 20])
bp = np.array([[0.5, 0.1], [0.05, 0.7]])
sbm_p_mat = sbm_p_matrix(m, bp)
dc_mat = np.outer(dc, dc)
heatmap(dc_mat)
heatmap(sbm_p_mat)
#%%
a = np.array([[0, 1], [0, 1]])
b = np.arange(0, 20).reshape((2, 10))
a
b
b[1, a]
#%%
