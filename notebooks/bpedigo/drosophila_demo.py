#%%
from graspy.models import EREstimator, SBEstimator, RDPGEstimator
from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.plot import heatmap
from graspy.utils import symmetrize, binarize
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ##################
# plot_graphs = False
# min_comp = 0
# max_comp = 2
# n_comp = 10
# ##################

## Load data
sns.set_context("talk")
left_adj, cell_labels = load_drosophila_right(return_labels=True)
left_adj_uw = left_adj.copy()
left_adj_uw[left_adj_uw > 0] = 1

left_adj_uw = symmetrize(left_adj_uw, method="avg")
left_adj_uw = binarize(left_adj_uw)


def evaluate_models(
    graph, labels=None, plot_graphs=False, min_comp=0, max_comp=1, n_comp=5
):

    if plot_graphs:
        heatmap(graph, inner_hier_labels=cell_labels)

    ## Set up models to test
    non_rdpg_models = [
        EREstimator(fit_degrees=False),
        EREstimator(fit_degrees=True),
        SBEstimator(fit_degrees=False),
        SBEstimator(fit_degrees=True),
    ]

    d = [int(i) for i in np.logspace(min_comp, max_comp, n_comp)]
    rdpg_models = [RDPGEstimator(n_components=i) for i in d]
    models = non_rdpg_models + rdpg_models

    names_nonRDPG = ["ER", "DCER", "SBM", "DCSBM"]
    names_RDPG = ["RDPG {}".format(i) for i in d]
    names = names_nonRDPG + names_RDPG

    bics = []
    log_likelihoods = []

    ## Test models
    for model, name in zip(models, names):
        m = model.fit(graph, y=labels)
        if plot_graphs:
            heatmap(m.p_mat_, inner_hier_labels=labels, title=(name + "P matrix"))
            heatmap(m.sample(), inner_hier_labels=labels, title=(name + "sample"))
        bic = m.bic(graph)
        log_likelihoods.append(m.score(graph))
        bics.append(bic)
        plt.show()

    bics = np.array(bics)
    log_likelihoods = np.array(log_likelihoods)

    ## Plot results
    plt.figure()
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
    sns.pointplot(names_nonRDPG, bics[:4], join=False, ax=ax[0])
    sns.scatterplot(d, bics[4:])
    ax[1].set_xlabel("RDPG - d")
    ax[0].set_xlabel("A priori models")
    ax[0].set_ylabel("rBIC")
    plt.suptitle("Drosophila left MB", y=0.94)

    plt.figure()
    fig, ax = plt.subplots(1, 2, sharey=True, figsize=(10, 10))
    sns.pointplot(names_nonRDPG, -log_likelihoods[:4], join=False, ax=ax[0])
    sns.scatterplot(d, -log_likelihoods[4:])
    ax[1].set_xlabel("RDPG - d")
    ax[0].set_xlabel("A priori models")
    ax[0].set_ylabel("-ln(Likelihood)")
    plt.suptitle("Drosophila left MB", y=0.94)

    return bics, log_likelihoods


evaluate_models(left_adj, labels=cell_labels, plot_graphs=True)
