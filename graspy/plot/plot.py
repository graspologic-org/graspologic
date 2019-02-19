# plot.py
# Created by Jaewon Chung on 2018-10-19.
# Email: j1c@jhu.edu
# Copyright (c) 2018. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from ..utils import import_graph, pass_to_ranks
from ..embed import selectSVD
from sklearn.utils import check_array, check_consistent_length
from mpl_toolkits.axes_grid1 import make_axes_locatable


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


def _transform(arr, method):
    if method is not None:
        if method == "log":
            # arr = np.log(arr, where=(arr > 0))
            # hacky, but np.log(arr, where=arr>0) is really buggy
            arr = arr.copy()
            arr[arr > 0] = np.log(arr[arr > 0])
        elif method in ["zero-boost", "simple-all", "simple-nonzero"]:
            arr = pass_to_ranks(arr, method=method)
        else:
            msg = "Transform must be one of {log, zero-boost, simple-all, \
            simple-nonzero, not {}.".format(
                method
            )
            raise ValueError(msg)

    return arr


def heatmap(
    X,
    transform=None,
    figsize=(10, 10),
    title=None,
    context="talk",
    font_scale=1,
    xticklabels=False,
    yticklabels=False,
    cmap="RdBu_r",
    center=0,
    cbar=True,
    inner_hier_labels=None,
    outer_hier_labels=None,
):
    r"""
    Plots a graph as a heatmap.

    Parameters
    ----------
    X : nx.Graph or np.ndarray object
        Graph or numpy matrix to plot
    transform : None, or string {'log', 'zero-boost', 'simple-all', 'simple-nonzero'}

        - 'log' :
            Plots the log of all nonzero numbers
        - 'zero-boost' :s
            Pass to ranks method. preserves the edge weight for all 0s, but ranks 
            the other edges as if the ranks of all 0 edges has been assigned. 
        - 'simple-all': 
            Pass to ranks method. Assigns ranks to all non-zero edges, settling 
            ties using the average. Ranks are then scaled by 
            :math:`\frac{2 rank(\text{non-zero edges})}{n^2 + 1}` 
            where n is the number of nodes
        - 'simple-nonzero':
            Pass to ranks method. Aame as simple-all, but ranks are scaled by
            :math:`\frac{2 rank(\text{non-zero edges})}{\text{total non-zero edges} + 1}`
    figsize : tuple of integers, optional, default: (10, 10)
        Width, height in inches.
    title : str, optional, default: None
        Title of plot.
    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font
        elements.
    xticklabels, yticklabels : bool or list, optional
        If list-like, plot these alternate labels as the ticklabels.
    cmap : str, default: 'RdBu_r'
        Valid matplotlib color map.
    center : float, default: 0
        The value at which to center the colormap
    cbar : bool, default: True
        Whether to draw a colorbar.
    inner_hier_labels : array-like, length of X's first dimension, default: None
        Categorical labeling of the nodes. If not None, will group the nodes 
        according to these labels and plot the labels on the marginal
    outer_hier_labels : array-like, length of X's first dimension, default: None
        Categorical labeling of the nodes, ignored without `inner_hier_labels`
        If not None, will plot these labels as the second level of a hierarchy on the
        marginals 
    """
    _check_common_inputs(
        figsize=figsize, title=title, context=context, font_scale=font_scale
    )

    # Handle ticklabels
    if isinstance(xticklabels, list):
        if len(xticklabels) != X.shape[1]:
            msg = "xticklabels must have same length {}.".format(X.shape[1])
            raise ValueError(msg)
    elif not isinstance(xticklabels, bool):
        msg = "xticklabels must be a bool or a list, not {}".format(type(xticklabels))
        raise TypeError(msg)

    if isinstance(yticklabels, list):
        if len(yticklabels) != X.shape[0]:
            msg = "yticklabels must have same length {}.".format(X.shape[0])
            raise ValueError(msg)
    elif not isinstance(yticklabels, bool):
        msg = "yticklabels must be a bool or a list, not {}".format(type(yticklabels))
        raise TypeError(msg)

    # Handle cmap
    if not isinstance(cmap, str):
        msg = "cmap must be a string, not {}.".format(type(cmap))
        raise TypeError(msg)

    # Handle center
    if center is not None:
        if not isinstance(center, (int, float)):
            msg = "center must be a integer or float, not {}.".format(type(center))
            raise TypeError(msg)

    # Handle cbar
    if not isinstance(cbar, bool):
        msg = "cbar must be a bool, not {}.".format(type(center))
        raise TypeError(msg)

    check_consistent_length(X, inner_hier_labels, outer_hier_labels)

    arr = import_graph(X)
    arr = _transform(arr, transform)
    if inner_hier_labels is not None:
        if outer_hier_labels is None:
            arr = _sort_graph(arr, inner_hier_labels, np.ones_like(inner_hier_labels))
        else:
            arr = _sort_graph(arr, inner_hier_labels, outer_hier_labels)

    # Global plotting settings
    CBAR_KWS = dict(shrink=0.7)

    with sns.plotting_context(context, font_scale=font_scale):
        fig, ax = plt.subplots(figsize=figsize)
        plot = sns.heatmap(
            arr,
            cmap=cmap,
            square=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cbar_kws=CBAR_KWS,
            center=center,
            cbar=cbar,
            ax=ax,
        )
        if title is not None:
            plot.set_title(title)
        if inner_hier_labels is not None:
            if outer_hier_labels is not None:
                plot.set_yticklabels([])
                plot.set_xticklabels([])
                _plot_groups(
                    plot, arr[0].shape[0], inner_hier_labels, outer_hier_labels
                )
            else:
                _plot_groups(plot, arr[0].shape[0], inner_hier_labels)
    return plot


def gridplot(
    X,
    labels=None,
    transform=None,
    height=10,
    title=None,
    context="talk",
    font_scale=1,
    alpha=0.7,
    sizes=(10, 200),
    palette="Set1",
    legend_name="Type",
    inner_hier_labels=None,
    outer_hier_labels=None,
):
    r"""
    Plots multiple graphs as a grid, with intensity denoted by the size 
    of dots on the grid.

    Parameters
    ----------
    X : list of nx.Graph or np.ndarray object
        List of nx.Graph or numpy arrays to plot
    labels : list of str
        List of strings, which are labels for each element in X. 
        `len(X) == len(labels)`.
    transform : None, or string {'log', 'zero-boost', 'simple-all', 'simple-nonzero'}

        - 'log' :
            Plots the log of all nonzero numbers
        - 'zero-boost' :
            Pass to ranks method. preserves the edge weight for all 0s, but ranks 
            the other edges as if the ranks of all 0 edges has been assigned. 
        - 'simple-all': 
            Pass to ranks method. Assigns ranks to all non-zero edges, settling 
            ties using the average. Ranks are then scaled by 
            :math:`\frac{2 rank(\text{non-zero edges})}{n^2 + 1}` 
            where n is the number of nodes
        - 'simple-nonzero':
            Pass to ranks method. Same as simple-all, but ranks are scaled by
            :math:`\frac{2 rank(\text{non-zero edges})}{\text{total non-zero edges} + 1}`
    height : int, optional, default: 10
        Height of figure in inches.
    title : str, optional, default: None
        Title of plot.
    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font
        elements.
    palette : str, dict, optional, default: 'Set1'
        Set of colors for mapping the `hue` variable. If a dict, keys should
        be values in the hue variable
    alpha : float [0, 1], default : 0.7
        alpha value of plotted gridplot points
    sizes : length 2 tuple, default: (10, 200)
        min and max size to plot edge weights
    legend_name : string, default: 'Type'
        Name to plot above the legend
    inner_hier_labels : array-like, length of X's first dimension, default: None
        Categorical labeling of the nodes. If not None, will group the nodes 
        according to these labels and plot the labels on the marginal
    outer_hier_labels : array-like, length of X's first dimension, default: None
        Categorical labeling of the nodes, ignored without `inner_hier_labels`
        If not None, will plot these labels as the second level of a hierarchy on the
        marginals
    """
    _check_common_inputs(
        height=height, title=title, context=context, font_scale=font_scale
    )

    if isinstance(X, list):
        graphs = [import_graph(x) for x in X]
    else:
        msg = "X must be a list, not {}.".format(type(X))
        raise TypeError(msg)

    check_consistent_length(X, labels, inner_hier_labels, outer_hier_labels)

    graphs = [_transform(arr, transform) for arr in graphs]

    if inner_hier_labels is not None:
        if outer_hier_labels is None:
            graphs = [
                _sort_graph(arr, inner_hier_labels, np.ones_like(inner_hier_labels))
                for arr in graphs
            ]
        else:
            graphs = [
                _sort_graph(arr, inner_hier_labels, outer_hier_labels) for arr in graphs
            ]

    if isinstance(palette, str):
        palette = sns.color_palette(palette, desat=0.75, n_colors=len(labels))

    dfs = []
    for idx, graph in enumerate(graphs):
        rdx, cdx = np.where(graph > 0)
        weights = graph[(rdx, cdx)]
        df = pd.DataFrame(
            np.vstack([rdx + 0.5, cdx + 0.5, weights]).T,
            columns=["rdx", "cdx", "Weights"],
        )
        df[legend_name] = [labels[idx]] * len(cdx)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    with sns.plotting_context(context, font_scale=font_scale):
        sns.set_style("white")
        plot = sns.relplot(
            data=df,
            x="cdx",
            y="rdx",
            hue=legend_name,
            size="Weights",
            sizes=sizes,
            alpha=alpha,
            palette=palette,
            height=height,
            facet_kws={
                "sharex": True,
                "sharey": True,
                "xlim": (0, graph.shape[0] + 1),
                "ylim": (0, graph.shape[0] + 1),
            },
        )
        plot.ax.axis("off")
        plot.ax.invert_yaxis()
        if title is not None:
            plot.set(title=title)
    if inner_hier_labels is not None:
        if outer_hier_labels is not None:
            _plot_groups(
                plot.ax, graphs[0].shape[0], inner_hier_labels, outer_hier_labels
            )
        else:
            _plot_groups(plot.ax, graphs[0].shape[0], inner_hier_labels)
    return plot


# TODO would it be cool if pairplot reduced to single plot
def pairplot(
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
):
    r"""
    Plot pairwise relationships in a dataset.

    Parameters
    ----------
    X : array-like, shape (n_samples, n_features)
        Input data.
    Y : array-like or list, shape (n_samples), optional
        Labels that correspond to each sample in X.
    col_names : array-like or list, shape (n_features), optional
        Names or labels for each feature in X. If not provided, the default 
        will be `Dimension 1, Dimension 2, etc`.
    title : str, optional, default: None
        Title of plot.
    legend_name : str, optional, default: None
        Title of the legend.
    variables : list of variable names, optional
        Variables to plot based on col_names, otherwise use every column with
        a numeric datatype.
    height : int, optional, default: 10
        Height of figure in inches.
    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font 
        elements.
    palette : str, dict, optional, default: 'Set1'
        Set of colors for mapping the `hue` variable. If a dict, keys should
        be values in the hue variable.
    alpha : float, optional, default: 0.7
        opacity value of plotter markers between 0 and 1 
    size : float or int, optional, default: 50
        size of plotted markers 
    marker : string, optional, default: '.'
        matplotlib style marker specification 
        https://matplotlib.org/api/markers_api.html
    """
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
            pairs = sns.pairplot(
                df,
                hue=legend_name,
                vars=variables,
                height=height,
                palette=palette,
                diag_kind=diag_kind,
                plot_kws=plot_kws,
            )
        else:
            pairs = sns.pairplot(
                df,
                vars=variables,
                height=height,
                palette=palette,
                diag_kind=diag_kind,
                plot_kws=plot_kws,
            )
        pairs.set(xticks=[], yticks=[])
        pairs.fig.subplots_adjust(top=0.945)
        pairs.fig.suptitle(title)

    return pairs


def _distplot(
    data,
    labels=None,
    direction="out",
    title="",
    context="talk",
    font_scale=1,
    figsize=(10, 5),
    palette="Set1",
    xlabel="",
    ylabel="Density",
):

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    palette = sns.color_palette(palette)
    plt_kws = {"cumulative": True}
    with sns.plotting_context(context=context, font_scale=font_scale):
        if labels is not None:
            categories, counts = np.unique(labels, return_counts=True)
            for i, cat in enumerate(categories):
                cat_data = data[np.where(labels == cat)]
                if counts[i] > 1 and cat_data.min() != cat_data.max():
                    x = np.sort(cat_data)
                    y = np.arange(len(x)) / float(len(x))
                    plt.plot(x, y, label=cat, color=palette[i])
                else:
                    ax.axvline(cat_data[0], label=cat, color=palette[i])
            plt.legend()
        else:
            if data.min() != data.max():
                sns.distplot(data, hist=False, kde_kws=plt_kws)
            else:
                ax.axvline(data[0])

        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)

    return ax


def degreeplot(
    X,
    labels=None,
    direction="out",
    title="Degree plot",
    context="talk",
    font_scale=1,
    figsize=(10, 5),
    palette="Set1",
):
    r"""
    Plots the distribution of node degrees for the input graph. 
    Allows for sets of node labels, will plot a distribution for each 
    node category. 
    
    Parameters
    ----------
    X : np.ndarray (2D)
        input graph 
    labels : 1d np.ndarray or list, same length as dimensions of X
        labels for different categories of graph nodes
    direction : string, ('out', 'in')
        for a directed graph, whether to plot out degree or in degree
    title : string, default : 'Degree plot'
        plot title 
    context :  None, or one of {talk (default), paper, notebook, poster}
        Seaborn plotting context
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font 
        elements.
    palette : str, dict, optional, default: 'Set1'
        Set of colors for mapping the `hue` variable. If a dict, keys should
        be values in the hue variable.
    figsize : tuple of length 2, default (10, 5)
        size of the figure (width, height)

    Returns 
    ------- 
    ax : matplotlib axis object
    """
    _check_common_inputs(
        figsize=figsize, title=title, context=context, font_scale=font_scale
    )
    check_array(X)
    if direction == "out":
        axis = 0
        check_consistent_length((X, labels))
    elif direction == "in":
        axis = 1
        check_consistent_length((X.T, labels))
    else:
        raise ValueError('direction must be either "out" or "in"')
    degrees = np.count_nonzero(X, axis=axis)
    ax = _distplot(
        degrees,
        labels=labels,
        title=title,
        context=context,
        font_scale=font_scale,
        figsize=figsize,
        palette=palette,
        xlabel="Node degree",
    )
    return ax


def edgeplot(
    X,
    labels=None,
    nonzero=False,
    title="Edge plot",
    context="talk",
    font_scale=1,
    figsize=(10, 5),
    palette="Set1",
):
    r"""
    Plots the distribution of edge weights for the input graph. 
    Allows for sets of node labels, will plot edge weight distribution 
    for each node category. 
    
    Parameters
    ----------
    X : np.ndarray (2D)
        input graph 
    labels : 1d np.ndarray or list, same length as dimensions of X
        labels for different categories of graph nodes
    nonzero : boolean, default: False
        whether to restrict the edgeplot to only the non-zero edges
    title : string, default : 'Degree plot'
        plot title 
    context :  None, or one of {talk (default), paper, notebook, poster}
        Seaborn plotting context
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font 
        elements.
    palette : str, dict, optional, default: 'Set1'
        Set of colors for mapping the `hue` variable. If a dict, keys should
        be values in the hue variable.
    figsize : tuple of length 2, default (10, 5)
        size of the figure (width, height)
        
    Returns 
    ------- 
    ax : matplotlib axis object
    """
    _check_common_inputs(
        figsize=figsize, title=title, context=context, font_scale=font_scale
    )
    check_array(X)
    check_consistent_length((X, labels))
    edges = X.ravel()
    labels = np.tile(labels, (1, X.shape[1]))
    labels = labels.ravel()
    if nonzero:
        labels = labels[edges != 0]
        edges = edges[edges != 0]
    ax = _distplot(
        edges,
        labels=labels,
        title=title,
        context=context,
        font_scale=font_scale,
        figsize=figsize,
        palette=palette,
        xlabel="Edge weight",
    )
    return ax


def screeplot(
    X,
    title="Scree plot",
    context="talk",
    font_scale=1,
    figsize=(10, 5),
    cumulative=True,
    show_first=None,
):
    r"""
    Plots the distribution of singular values for a matrix, either showing the 
    raw distribution or an empirical CDF (depending on `cumulative`)

    Parameters
    ----------
    X : np.ndarray (2D)
        input matrix 
    title : string, default : 'Degree plot'
        plot title 
    context :  None, or one of {talk (default), paper, notebook, poster}
        Seaborn plotting context
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font 
        elements.
    figsize : tuple of length 2, default (10, 5)
        size of the figure (width, height)
    cumulative : boolean, default: True
        whether or not to plot a cumulative cdf of singular values 
    show_first : int or None, default: None 
        whether to restrict the plot to the first `show_first` components

    Returns
    -------
    ax : matplotlib axis object
    """
    _check_common_inputs(
        figsize=figsize, title=title, context=context, font_scale=font_scale
    )
    check_array(X)
    if show_first is not None:
        if not isinstance(show_first, int):
            msg = "show_first must be an int"
            raise TypeError(msg)
    if not isinstance(cumulative, bool):
        msg = "cumulative must be a boolean"
        raise TypeError(msg)
    _, D, _ = selectSVD(X, n_components=X.shape[1], algorithm="full")
    D /= D.sum()
    if cumulative:
        y = np.cumsum(D[:show_first])
    else:
        y = D[:show_first]
    _ = plt.figure(figsize=figsize)
    ax = plt.gca()
    xlabel = "Component"
    ylabel = "Variance explained"
    with sns.plotting_context(context=context, font_scale=font_scale):
        plt.plot(y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    return ax


def _sort_inds(inner_labels, outer_labels):
    sort_df = pd.DataFrame(columns=("inner_labels", "outer_labels"))
    sort_df["inner_labels"] = inner_labels
    if outer_labels is not None:
        sort_df["outer_labels"] = outer_labels
        sort_df.sort_values(
            by=["outer_labels", "inner_labels"], kind="mergesort", inplace=True
        )
        outer_labels = sort_df["outer_labels"]
    inner_labels = sort_df["inner_labels"]
    sorted_inds = sort_df.index.values
    return sorted_inds


def _sort_graph(graph, inner_labels, outer_labels):
    inds = _sort_inds(inner_labels, outer_labels)
    graph = graph[inds, :][:, inds]
    return graph


def _get_freqs(inner_labels, outer_labels=None):
    _, outer_freq = np.unique(outer_labels, return_counts=True)
    outer_freq_cumsum = np.hstack((0, outer_freq.cumsum()))

    # for each group of outer labels, calculate the boundaries of the inner labels
    inner_freq = np.array([])
    for i in range(outer_freq.size):
        start_ind = outer_freq_cumsum[i]
        stop_ind = outer_freq_cumsum[i + 1]
        _, temp_freq = np.unique(inner_labels[start_ind:stop_ind], return_counts=True)
        inner_freq = np.hstack([inner_freq, temp_freq])
    inner_freq_cumsum = np.hstack((0, inner_freq.cumsum()))

    return inner_freq, inner_freq_cumsum, outer_freq, outer_freq_cumsum


# assume that the graph has already been plotted in sorted form
def _plot_groups(ax, n_verts, inner_labels, outer_labels=None):
    plot_outer = True
    if outer_labels is None:
        outer_labels = np.ones_like(inner_labels)
        plot_outer = False
    sorted_inds = _sort_inds(inner_labels, outer_labels)
    inner_labels = inner_labels[sorted_inds]
    outer_labels = outer_labels[sorted_inds]
    inner_freq, inner_freq_cumsum, outer_freq, outer_freq_cumsum = _get_freqs(
        inner_labels, outer_labels
    )

    inner_unique = np.unique(inner_labels)
    outer_unique = np.unique(outer_labels)

    # draw lines
    for x in inner_freq_cumsum:
        ax.vlines(x, 0, n_verts, linestyle="dashed", lw=0.9, alpha=0.25, zorder=3)
        if x == inner_freq_cumsum[-1]:
            x -= 1
        ax.hlines(x, 0, n_verts, linestyle="dashed", lw=0.9, alpha=0.25, zorder=3)

    # generic curve that we will use for everything
    lx = np.linspace(-np.pi / 2.0 + 0.05, np.pi / 2.0 - 0.05, 50)
    tan = np.tan(lx)
    curve = np.hstack((tan[::-1], tan))

    divider = make_axes_locatable(ax)

    # inner curve generation
    inner_tick_loc = inner_freq.cumsum() - inner_freq / 2
    inner_tick_width = inner_freq / 2
    # outer curve generation
    outer_tick_loc = outer_freq.cumsum() - outer_freq / 2
    outer_tick_width = outer_freq / 2

    # top inner curves
    # ax_x = divider.new_vertical(
    #     size="5%", pad=0.0, sharex=ax, pack_start=False)
    ax_x = divider.new_vertical(size="5%", pad=0.0, pack_start=False)
    ax.figure.add_axes(ax_x)
    _plot_brackets(
        ax_x,
        np.tile(inner_unique, len(outer_unique)),
        inner_tick_loc,
        inner_tick_width,
        curve,
        "inner",
        "x",
        n_verts,
    )
    # side inner curves
    # ax_y = divider.new_horizontal(
    #     size="5%", pad=0.0, sharey=ax, pack_start=True)
    ax_y = divider.new_horizontal(size="5%", pad=0.0, pack_start=True)
    ax.figure.add_axes(ax_y)
    _plot_brackets(
        ax_y,
        np.tile(inner_unique, len(outer_unique)),
        inner_tick_loc,
        inner_tick_width,
        curve,
        "inner",
        "y",
        n_verts,
    )

    if plot_outer:
        # top outer curves
        ax_x2 = divider.new_vertical(size="5%", pad=0.25, pack_start=False)
        ax.figure.add_axes(ax_x2)
        _plot_brackets(
            ax_x2,
            outer_unique,
            outer_tick_loc,
            outer_tick_width,
            curve,
            "outer",
            "x",
            n_verts,
        )
        # side outer curves
        ax_y2 = divider.new_horizontal(size="5%", pad=0.25, pack_start=True)
        ax.figure.add_axes(ax_y2)
        _plot_brackets(
            ax_y2,
            outer_unique,
            outer_tick_loc,
            outer_tick_width,
            curve,
            "outer",
            "y",
            n_verts,
        )
    return ax


def _plot_brackets(ax, group_names, tick_loc, tick_width, curve, level, axis, max_size):
    for x0, width in zip(tick_loc, tick_width):
        x = np.linspace(x0 - width, x0 + width, 100)
        if axis == "x":
            ax.plot(x, -curve, c="k")
        elif axis == "y":
            ax.plot(curve, x, c="k")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.tick_params(axis=axis, which=u"both", length=0, pad=7)
    for direction in ["left", "right", "bottom", "top"]:
        ax.spines[direction].set_visible(False)
    if axis == "x":
        ax.set_xticks(tick_loc)
        ax.set_xticklabels(group_names, fontsize=15, verticalalignment="center")
        ax.xaxis.set_label_position("top")
        ax.xaxis.tick_top()
        ax.set_xlim(0, max_size)
    elif axis == "y":
        ax.set_yticks(tick_loc)
        ax.set_yticklabels(group_names, fontsize=15, verticalalignment="center")
        # ax.yaxis.set_label_position('top')
        # ax.yaxis.tick_top()
        ax.set_ylim(0, max_size)
        ax.invert_yaxis()
