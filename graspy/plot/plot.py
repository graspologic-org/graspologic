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


def _check_common_inputs(figsize=None,
                         height=None,
                         title=None,
                         context=None,
                         font_scale=None,
                         legend_name=None):
    # Handle figsize
    if figsize is not None:
        if not isinstance(figsize, tuple):
            msg = 'figsize must be a tuple, not {}.'.format(type(figsize))
            raise TypeError(msg)

    # Handle heights
    if height is not None:
        if not isinstance(height, (int, float)):
            msg = 'height must be an integer or float, not {}.'.format(
                type(height))
            raise TypeError(msg)

    # Handle title
    if title is not None:
        if not isinstance(title, str):
            msg = 'title must be a string, not {}.'.format(type(title))
            raise TypeError(msg)

    # Handle context
    if context is not None:
        if not isinstance(context, str):
            msg = 'context must be a string, not {}.'.format(type(context))
            raise TypeError(msg)
        elif not context in ['paper', 'notebook', 'talk', 'poster']:
            msg = 'context must be one of (paper, notebook, talk, poster), \
                not {}.'.format(context)
            raise ValueError(msg)

    # Handle font_scale
    if font_scale is not None:
        if not isinstance(font_scale, (int, float)):
            msg = 'font_scale must be an integer or float, not {}.'.format(
                type(font_scale))
            raise TypeError(msg)

    # Handle legend name
    if legend_name is not None:
        if not isinstance(legend_name, str):
            msg = 'legend_name must be a string, not {}.'.format(
                type(legend_name))
            raise TypeError(msg)


def _transform(arr, method):
    if method is not None:
        if method == 'log':
            #arr = np.log(arr, where=(arr > 0))
            #hacky, but np.log(arr, where=arr>0) is really buggy
            arr = arr.copy()
            arr[arr > 0] = np.log(arr[arr > 0])
        elif method in ['zero-boost', 'simple-all', 'simple-nonzero']:
            arr = pass_to_ranks(arr, method=method)
        else:
            msg = 'Transform must be one of {log, zero-boost, simple-all, \
            simple-nonzero, not {}.'.format(method)
            raise ValueError(msg)

    return arr


def heatmap(X,
            transform=None,
            figsize=(10, 10),
            title=None,
            context='talk',
            font_scale=1,
            xticklabels=False,
            yticklabels=False,
            cmap='RdBu_r',
            center=0,
            cbar=True):
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
    """
    _check_common_inputs(
        figsize=figsize, title=title, context=context, font_scale=font_scale)

    # Handle ticklabels
    if isinstance(xticklabels, list):
        if len(xticklabels) != X.shape[1]:
            msg = 'xticklabels must have same length {}.'.format(X.shape[1])
            raise ValueError(msg)
    elif not isinstance(xticklabels, bool):
        msg = 'xticklabels must be a bool or a list, not {}'.format(
            type(xticklabels))
        raise TypeError(msg)

    if isinstance(yticklabels, list):
        if len(yticklabels) != X.shape[0]:
            msg = 'yticklabels must have same length {}.'.format(X.shape[0])
            raise ValueError(msg)
    elif not isinstance(yticklabels, bool):
        msg = 'yticklabels must be a bool or a list, not {}'.format(
            type(yticklabels))
        raise TypeError(msg)

    # Handle cmap
    if not isinstance(cmap, str):
        msg = 'cmap must be a string, not {}.'.format(type(cmap))
        raise TypeError(msg)

    # Handle center
    if center is not None:
        if not isinstance(center, (int, float)):
            msg = 'center must be a integer or float, not {}.'.format(
                type(center))
            raise TypeError(msg)

    # Handle cbar
    if not isinstance(cbar, bool):
        msg = 'cbar must be a bool, not {}.'.format(type(center))
        raise TypeError(msg)

    arr = import_graph(X)
    arr = _transform(arr, transform)
    #arr = _transform(X, transform)

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
            ax=ax)
        if title is not None:
            plot.set_title(title)

    return plot


def gridplot(X,
             labels=None,
             transform=None,
             height=10,
             title=None,
             context='talk',
             font_scale=1,
             alpha=0.7,
             sizes=(10, 200),
             palette='Set1',
             legend_name='Type'):
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
    legend_name : string
    """
    _check_common_inputs(
        height=height, title=title, context=context, font_scale=font_scale)

    if isinstance(X, list):
        graphs = [import_graph(x) for x in X]
    else:
        msg = 'X must be a list, not {}.'.format(type(X))
        raise TypeError(msg)

    # Handle labels
    if not isinstance(labels, list):
        msg = 'labels must be a list, not {}.'.format(type(labels))
        raise TypeError(msg)
    elif len(labels) != len(graphs):
        msg = 'Expected {} elements in labels, but got {} instead.'.format(
            len(graphs), len(labels))
        raise ValueError(msg)

    graphs = [_transform(arr, transform) for arr in graphs]

    # palette = sns.color_palette('Set1', desat=0.75, n_colors=len(labels))

    dfs = []
    for idx, graph in enumerate(graphs):
        rdx, cdx = np.where(graph > 0)
        weights = graph[(rdx, cdx)]
        df = pd.DataFrame(
            np.vstack([rdx + 0.5, cdx + 0.5, weights]).T,
            columns=['rdx', 'cdx', 'Weights'])
        df[legend_name] = [labels[idx]] * len(cdx)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    with sns.plotting_context(context, font_scale=font_scale):
        sns.set_style('white')
        plot = sns.relplot(
            data=df,
            x='cdx',
            y='rdx',
            hue=legend_name,
            size='Weights',
            sizes=sizes,
            alpha=alpha,
            palette=palette,
            height=height,
            facet_kws={
                'sharex': True,
                'sharey': True,
                'xlim': (0, graph.shape[0] + 1),
                'ylim': (0, graph.shape[0] + 1),
            },
        )
        plot.ax.axis('off')
        plot.ax.invert_yaxis()
        if title is not None:
            plot.set(title=title)

    return plot


# TODO would it be cool if pairplot reduced to single plot
def pairplot(X,
             labels=None,
             col_names=None,
             title=None,
             legend_name=None,
             variables=None,
             height=2.5,
             context='talk',
             font_scale=1,
             palette='Set1',
             alpha=0.7,
             size=50,
             marker='.'):
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
        legend_name=legend_name)

    # Handle X
    if not isinstance(X, (list, np.ndarray)):
        msg = 'X must be array-like, not {}.'.format(type(X))
        raise TypeError(msg)

    # Handle Y
    if labels is not None:
        if not isinstance(labels, (list, np.ndarray)):
            msg = 'Y must be array-like or list, not {}.'.format(type(labels))
            raise TypeError(msg)
        elif X.shape[0] != len(labels):
            msg = 'Expected length {}, but got length {} instead for Y.'.format(
                X.shape[0], len(labels))
            raise ValueError(msg)

    # Handle col_names
    if col_names is None:
        col_names = [
            'Dimension {}'.format(i) for i in range(1, X.shape[1] + 1)
        ]
    elif not isinstance(col_names, list):
        msg = 'col_names must be a list, not {}.'.format(type(col_names))
        raise TypeError(msg)
    elif X.shape[1] != len(col_names):
        msg = 'Expected length {}, but got length {} instead for col_names.'.format(
            X.shape[1], len(col_names))
        raise ValueError(msg)

    # Handle variables
    if variables is not None:
        if len(variables) > len(col_names):
            msg = 'variables cannot contain more elements than col_names.'
            raise ValueError(msg)
        else:
            for v in variables:
                if v not in col_names:
                    msg = '{} is not a valid key.'.format(v)
                    raise KeyError(msg)
    else:
        variables = col_names

    diag_kind = 'auto'
    df = pd.DataFrame(X, columns=col_names)
    if labels is not None:
        if legend_name is None:
            legend_name = 'Type'
        df_labels = pd.DataFrame(labels, columns=[legend_name])
        df = pd.concat([df_labels, df], axis=1)

        names, counts = np.unique(labels, return_counts=True)
        if counts.min() < 2:
            diag_kind = 'hist'
    plot_kws = dict(
        alpha=alpha,
        s=size,
        # edgecolor=None, # could add this latter
        linewidth=0,
        marker=marker)
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


def _distplot(data,
              labels=None,
              direction='out',
              title='',
              context='talk',
              font_scale=1,
              figsize=(10, 5),
              palette='Set1',
              xlabel='',
              ylabel='Density'):

    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    palette = sns.color_palette(palette)
    plt_kws = {'cumulative': True}
    with sns.plotting_context(context=context, font_scale=font_scale):
        if labels is not None:
            categories, counts = np.unique(labels, return_counts=True)
            for i, cat in enumerate(categories):
                cat_data = data[np.where(labels == cat)]
                if counts[i] > 1 and cat_data.min() != cat_data.max():
                    x = np.sort(cat_data)
                    y = np.arange(len(x)) / float(len(x))
                    plt.plot(x, y, label=cat, color=palette[i])
                    # plt.plot(

                    #     np.cumsum(cat_data) / cat_data.sum(),
                    #     label=cat,
                    #     color=palette[i])
                    # sns.distplot(
                    #     cat_data,
                    #     label=cat,
                    #     hist=False,
                    #     color=palette[i],
                    #     kde_kws=plt_kws)
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


def degreeplot(X,
               labels=None,
               direction='out',
               title='Degree plot',
               context='talk',
               font_scale=1,
               figsize=(10, 5),
               palette='Set1'):
    if direction == 'out':
        axis = 0
    elif direction == 'in':
        axis = 1
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
        xlabel='Node degree')
    return ax


def edgeplot(X,
             labels=None,
             nonzero=False,
             title='Edge plot',
             context='talk',
             font_scale=1,
             figsize=(10, 5),
             palette='Set1'):
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
        xlabel='Edge weight')
    return ax


def screeplot(X,
              title='Scree plot',
              context='talk',
              font_scale=1,
              figsize=(10, 5),
              xlabel='Component',
              ylabel='Variance explained'):
    _, D, _ = selectSVD(X, n_components=X.shape[1], algorithm='full')
    D /= D.sum()
    x = np.sort(D)
    y = np.arange(len(x)) / float(len(x))
    fig = plt.figure(figsize=figsize)
    ax = plt.gca()
    with sns.plotting_context(context=context, font_scale=font_scale):
        plt.plot(x, y)
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    return ax


def gmmplot(means, covariances, ax=None):
    # TODO maybe
    # not sure how to implement, should it just be private method
    # called by pairplot?
    return 1


def grouped_heatmap():
    # TODO maybe
    return 1


def grouped_gridplot():
    # TODO maybe
    return 1
