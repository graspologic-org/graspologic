#%%
from graspy.utils import *
from graspy.plot import *
from graspy.datasets import load_drosophila_left, load_drosophila_right
from graspy.simulations import sample_edges, p_from_latent
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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
# pairplot(struct_latent)
# pairplot(xhat)
# pairplot(np.concatenate((xhat, means), axis=0))
struct_latent = means
for l in ["I", "O", "P"]:
    mean = np.mean(xhat[right_labels == l], axis=0)
    print(mean)
    n_cells = (right_labels == l).sum()
    latent_mat = np.zeros((n_cells, 6))
    for i, c in enumerate(xhat[right_labels == l]):
        proj = np.dot(c, mean)
        proj_vec = proj * mean
        latent_mat[i, :] = mean

    # latent_mat[:, :] = mean[np.newaxis, :]
    struct_latent = np.concatenate((struct_latent, latent_mat), axis=0)
struct_latent[:, 0] *= -1
scatterplot(struct_latent, labels=right_labels, font_scale=2)
scatterplot(xhat, labels=right_labels, font_scale=2)
p_mat = p_from_latent(
    struct_latent[:, :3], struct_latent[:, 3:], rescale=False, loops=False
)
graph = sample_edges(p_mat, directed=True, loops=False)
#%%
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


heatmap(graph, inner_hier_labels=right_labels)
heatmap(right_adj_raw, inner_hier_labels=right_labels, transform="zero-boost")
heatmap(p_mat, inner_hier_labels=right_labels)
from graspy.embed import AdjacencySpectralEmbed

l = AdjacencySpectralEmbed().fit_transform(p_mat)
l = np.concatenate((l[0], l[1]), axis=1)
scatterplot(l, labels=right_labels, font_scale=2)
plt.savefig("dcsbmlsm_latent.png", dpi=300)
heatmap(graph, inner_hier_labels=right_labels, cbar=False)
plt.savefig("dcsbmlsm_graph.png", dpi=300)
#%%

from graspy.models import SBEstimator
from graspy.embed import AdjacencySpectralEmbed

right_adj_bin = binarize(right_adj_raw)
sb = SBEstimator(fit_degrees=True, degree_directed=True)
sb.fit(right_adj_bin, right_labels)
p_mat = sb.p_mat_
heatmap(p_mat)
heatmap(right_adj_bin)
toy = right_adj_bin - p_mat
heatmap(toy)
ase = AdjacencySpectralEmbed()
X, Y = ase.fit_transform(toy)
latent = np.concatenate((X, Y), axis=1)
pairplot(latent, labels=right_labels, diag_kind="hist")

toy_p_mat = p_from_latent(X, Y, rescale=False) + p_mat
heatmap(toy_p_mat, inner_hier_labels=right_labels)

#%%
sb = SBEstimator(fit_degrees=True, degree_directed=True)
sb.fit(right_adj_bin, right_labels)
sb.bic(right_adj_bin)
sb.score(right_adj_bin)
heatmap(sb.sample(), cbar=False)
plt.savefig("ddcsbm_graph.png", dpi=300)
X, Y = AdjacencySpectralEmbed().fit_transform(sb.p_mat_)
latent = np.concatenate((X, Y), axis=1)
scatterplot(-latent, labels=right_labels)
sb = SBEstimator(fit_degrees=True, degree_directed=False)
sb.fit(right_adj_bin, right_labels)
sb.bic(right_adj_bin)
sb.score(right_adj_bin)
