#%%
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from graspy.datasets import load_drosophila_left
from graspy.embed import AdjacencySpectralEmbed
from graspy.models import EREstimator, RDPGEstimator, SBEstimator
from graspy.plot import heatmap, pairplot
import pandas as pd

#%% Set up some simulations
from graspy.simulations import p_from_latent, sample_edges
from graspy.utils import binarize, symmetrize

## Load data
sns.set_context("talk")
left_adj, cell_labels = load_drosophila_left(return_labels=True)
left_adj_uw = left_adj.copy()
left_adj_uw[left_adj_uw > 0] = 1

left_adj_uw = symmetrize(left_adj_uw, method="avg")
left_adj_uw = binarize(left_adj_uw)


def _check_common_inputs(
    figsize=None,
    height=None,
    title=None,
    context=None,
    font_scale=None,
    legend_name=None,
):
    # Handle figsize
    if figsize is not None:
        if not isinstance(figsize, tuple):
            msg = "figsize must be a tuple, not {}.".format(type(figsize))
            raise TypeError(msg)

    # Handle heights
    if height is not None:
        if not isinstance(height, (int, float)):
            msg = "height must be an integer or float, not {}.".format(type(height))
            raise TypeError(msg)

    # Handle title
    if title is not None:
        if not isinstance(title, str):
            msg = "title must be a string, not {}.".format(type(title))
            raise TypeError(msg)

    # Handle context
    if context is not None:
        if not isinstance(context, str):
            msg = "context must be a string, not {}.".format(type(context))
            raise TypeError(msg)
        elif not context in ["paper", "notebook", "talk", "poster"]:
            msg = "context must be one of (paper, notebook, talk, poster), \
                not {}.".format(
                context
            )
            raise ValueError(msg)

    # Handle font_scale
    if font_scale is not None:
        if not isinstance(font_scale, (int, float)):
            msg = "font_scale must be an integer or float, not {}.".format(
                type(font_scale)
            )
            raise TypeError(msg)

    # Handle legend name
    if legend_name is not None:
        if not isinstance(legend_name, str):
            msg = "legend_name must be a string, not {}.".format(type(legend_name))
            raise TypeError(msg)


def scatterplot(
    X,
    labels=None,
    col_names=None,
    title=None,
    legend_name=None,
    variables=None,
    height=2.5,
    context="talk",
    font_scale=1,
    palette="Set1",
    alpha=0.7,
    size=50,
    marker=".",
    legend=False,
):
    _check_common_inputs(
        height=height,
        title=title,
        context=context,
        font_scale=font_scale,
        legend_name=legend_name,
    )

    # Handle X
    if not isinstance(X, (list, np.ndarray)):
        msg = "X must be array-like, not {}.".format(type(X))
        raise TypeError(msg)

    # Handle Y
    if labels is not None:
        if not isinstance(labels, (list, np.ndarray)):
            msg = "Y must be array-like or list, not {}.".format(type(labels))
            raise TypeError(msg)
        elif X.shape[0] != len(labels):
            msg = "Expected length {}, but got length {} instead for Y.".format(
                X.shape[0], len(labels)
            )
            raise ValueError(msg)

    # Handle col_names
    if col_names is None:
        col_names = ["Dimension {}".format(i) for i in range(1, X.shape[1] + 1)]
    elif not isinstance(col_names, list):
        msg = "col_names must be a list, not {}.".format(type(col_names))
        raise TypeError(msg)
    elif X.shape[1] != len(col_names):
        msg = "Expected length {}, but got length {} instead for col_names.".format(
            X.shape[1], len(col_names)
        )
        raise ValueError(msg)

    # Handle variables
    if variables is not None:
        if len(variables) > len(col_names):
            msg = "variables cannot contain more elements than col_names."
            raise ValueError(msg)
        else:
            for v in variables:
                if v not in col_names:
                    msg = "{} is not a valid key.".format(v)
                    raise KeyError(msg)
    else:
        variables = col_names

    diag_kind = "auto"
    df = pd.DataFrame(X, columns=col_names)
    if labels is not None:
        if legend_name is None:
            legend_name = "Type"
        df_labels = pd.DataFrame(labels, columns=[legend_name])
        df = pd.concat([df_labels, df], axis=1)

        names, counts = np.unique(labels, return_counts=True)
        if counts.min() < 2:
            diag_kind = "hist"
    plot_kws = dict(
        alpha=alpha,
        s=size,
        # edgecolor=None, # could add this latter
        linewidth=0,
        marker=marker,
    )
    with sns.plotting_context(context=context, font_scale=font_scale):
        if labels is not None:
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            pairs = sns.scatterplot(
                data=df,
                x="Dimension 1",
                y="Dimension 2",
                hue=legend_name,
                # vars=variables,
                # height=height,
                palette=palette,
                # plot_kws=plot_kws,
                markers=marker,
                legend=False,
                s=size,
                alpha=alpha,
                linewidth=0,
                ax=ax,
            )
            pairs.axis("square")
        else:
            pairs = sns.scatterplot(
                df, vars=variables, height=height, palette=palette, plot_kws=plot_kws
            )
        pairs.set(xticks=[], yticks=[])
        # pairs.fig.subplots_adjust(top=0.945)
        # pairs.fig.suptitle(title)

    return pairs


def evaluate_models(
    graph, labels=None, title=None, plot_graphs=False, min_comp=0, max_comp=1, n_comp=5
):

    if plot_graphs:
        heatmap(graph, inner_hier_labels=cell_labels)

    ## Set up models to test
    non_rdpg_models = [
        EREstimator(fit_degrees=False),
        SBEstimator(fit_degrees=False),
        SBEstimator(fit_degrees=True),
    ]

    d = [6]
    rdpg_models = [RDPGEstimator(n_components=i) for i in d]
    models = non_rdpg_models + rdpg_models

    names_nonRDPG = ["ER", "SBM", "DCSBM"]
    names_RDPG = ["RDPGrank{}".format(i) for i in d]
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
        ase = AdjacencySpectralEmbed(n_components=2)
        latent = ase.fit_transform(m.p_mat_)
        # if type(latent) is tuple:
        #     pairplot(np.concatenate((latent[0], latent[1]), axis=1))
        #     plt.show()
        # else:
        print("here")
        # plt.figure(figsize=(20, 20))
        ax = scatterplot(
            latent, labels=cell_labels, height=4, alpha=0.6, font_scale=1.25
        )
        # plt.suptitle(name, y=0.94, x=0.1, fontsize=30, horizontalalignment="left")
        plt.savefig(name + "latent.png", format="png", dpi=1000)
        plt.close()


evaluate_models(left_adj_uw, labels=cell_labels)

# ax = heatmap(left_adj_uw, inner_hier_labels=cell_labels, cbar=False)
# ax2 = heatmap(left_adj_uw, inner_hier_labels=cell_labels, cbar=False)
