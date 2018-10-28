# plot.py
# Created by Jaewon Chung on 2018-10-19.
# Email: j1c@jhu.edu
# Copyright (c) 2018. All rights reserved.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..utils import import_graph, pass_to_ranks

# Global plotting settings
CBAR_KWS = dict(shrink=0.7)


def heatmap(X,
            transform=None,
            figsize=(10, 10),
            title=None,
            context='talk',
            font_scale=1,
            xticklabels=False,
            yticklabels=False,
            cmap='Reds',
            center=None,
            cbar=True):
    """
    Plots a graph as a heatmap.

    Parameters
    ----------
    X : nx.Graph or np.ndarray object
        Graph or numpy matrix to plot
    transform : None, or string {log, zero-boost, simple-all, simple-nonzero}
        log :
            Plots the log of all nonzero numbers
        zero-boost :
            Pass to ranks method. preserves the edge weight for all 0s, but ranks 
            the other edges as if the ranks of all 0 edges has been assigned. 
        'simple-all': 
            Pass to ranks method. Assigns ranks to all non-zero edges, settling 
            ties using the average. Ranks are then scaled by 
                .. math:: \frac{2 rank(non-zero edges)}{n^2 + 1}
            where n is the number of nodes
        'simple-nonzero':
            Pass to ranks method. Same as 'simple-all' but ranks are scaled by
                .. math:: \frac{2 rank(non-zero edges)}{num_nonzero + 1}
    figsize : tuple of integers, optional, default: (10, 10)
        Width, height in inches.
    title : str, optional, default: None
        Title of plot.
    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font elements.
    xticklabels, yticklabels : bool or list, optional
        If list-like, plot these alternate labels as the ticklabels.
    cmap : str
        Valid color map.
    center : float, optional, default: None
        The value at which to center the colormap when plotting divergant data.
    cbar : bool, default: True
        Whether to draw a colorbar.
    """
    arr = import_graph(X)

    if transform is not None:
        if transform == 'log':
            #arr = np.log(arr, where=(arr > 0))
            #hacky, but np.log(arr, where=arr>0) is really buggy
            arr = arr.copy()
            arr[arr > 0] = np.log(arr[arr > 0])
        elif transform in ['zero-boost', 'simple-all', 'simple-nonzero']:
            arr = pass_to_ranks(arr, method=transform)
        else:
            msg = 'Transform must be one of {log, zero-boost, simple-all, \
            simple-nonzero, not {}.'.format(transform)
            raise ValueError(msg)

    # Handle figsize
    if not isinstance(figsize, tuple):
        msg = 'figsize must be a tuple, not {}.'.format(type(figsize))
        raise TypeError(msg)

    # Handle title
    if title is not None:
        if not isinstance(title, str):
            msg = 'title must be a string, not {}.'.format(type(title))
            raise TypeError(msg)

    # Handle context
    if not isinstance(context, str):
        msg = 'context must be a string, not {}.'.format(type(context))
        raise TypeError(msg)
    elif not context in ['paper', 'notebook', 'talk', 'poster']:
        msg = 'context must be one of (paper, notebook, talk, poster), \
            not {}.'.format(context)
        raise ValueError(msg)

    # Handle font_scale
    if not isinstance(font_scale, (int, float)):
        msg = 'font_scale must be an integer or float, not {}.'.format(
            type(font_scale))
        raise TypeError(msg)

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

    with sns.plotting_context(context, font_scale=font_scale):
        fig = plt.figure(figsize=figsize)
        plot = sns.heatmap(
            arr,
            cmap=cmap,
            square=True,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            cbar_kws=CBAR_KWS,
            center=center,
            cbar=cbar)
        if title is not None:
            plot.set_title(title)
        fig.tight_layout()

    return fig


def gridplot(X,
             labels,
             transform=None,
             height=10,
             title=None,
             context='talk',
             font_scale=1):
    """
    Plots multiple graphs as a grid, with intensity denoted by the size 
    of dots on the grid.

    Parameters
    ----------
    X : list of nx.Graph or np.ndarray object
        List of nx.Graph or numpy arrays to plot
    labels : list of str
        List of strings, which are labels for each element in X. 
        `len(X) == len(labels)`.
    transform : None, or string {log, zero-boost, simple-all, simple-nonzero}
        log :
            Plots the log of all nonzero numbers
        zero-boost :
            Pass to ranks method. preserves the edge weight for all 0s, but ranks 
            the other edges as if the ranks of all 0 edges has been assigned. 
        'simple-all': 
            Pass to ranks method. Assigns ranks to all non-zero edges, settling 
            ties using the average. Ranks are then scaled by 
                .. math:: \frac{2 rank(non-zero edges)}{n^2 + 1}
            where n is the number of nodes
        'simple-nonzero':
            Pass to ranks method. Same as 'simple-all' but ranks are scaled by
                .. math:: \frac{2 rank(non-zero edges)}{num_nonzero + 1}
    height : integers, optional, default: 10
        Height of figure in inches.
    title : str, optional, default: None
        Title of plot.
    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font elements.
    """
    if isinstance(X, list):
        graphs = [import_graph(x) for x in X]
    else:
        graphs = [import_graph(X)]

    # Handle labels
    if not isinstance(labels, list):
        msg = 'labels must be a list, not {}.'.format(type(labels))
        raise TypeError(msg)
    elif len(labels) != len(graphs):
        msg = 'Expected {} elements in labels, but got {} instead.'.format(
            len(graphs), len(labels))
        raise ValueError(msg)

    if transform is not None:
        if transform == 'log':
            #arr = np.log(arr, where=(arr > 0))
            #hacky, but np.log(arr, where=arr>0) is really buggy
            arr = arr.copy()
            arr[arr > 0] = np.log(arr[arr > 0])
        elif transform in ['zero-boost', 'simple-all', 'simple-nonzero']:
            arr = pass_to_ranks(arr, method=transform)
        else:
            msg = 'Transform must be one of {log, zero-boost, simple-all, \
            simple-nonzero, not {}.'.format(transform)
            raise ValueError(msg)

    # Handle heights
    if not isinstance(height, (int, float)):
        msg = 'height must be an integer or float, not {}.'.format(
            type(height))
        raise TypeError(msg)

    # Handle title
    if not isinstance(title, str):
        msg = 'title must be a string, not {}.'.format(type(title))
        raise TypeError(msg)

    # Handle context
    if not context in ['paper', 'notebook', 'talk', 'poster']:
        msg = 'context must be one of {paper, notebook, talk, poster}, \
            not {}.'.format(context)
        raise ValueError(msg)

    # Handle font_scale
    if not isinstance(font_scale, (int, float)):
        msg = 'font_scale must be an integer or float, not {}.'.format(
            type(font_scale))
        raise TypeError(msg)

    palette = sns.color_palette('Set1', desat=0.75, n_colors=len(labels))

    dfs = []
    for idx, graph in enumerate(graphs):
        cdx, rdx = np.where(graph > 0)
        weights = graph[(cdx, rdx)]
        df = pd.DataFrame(
            np.vstack([cdx[::-1], rdx, weights]).T,
            columns=['cdx', 'rdx', 'Weights'])
        df['Type'] = [labels[idx]] * len(cdx)
        dfs.append(df)

    df = pd.concat(dfs, axis=0)

    with sns.plotting_context(context, font_scale=font_scale):
        plot = sns.relplot(
            data=df,
            x='cdx',
            y='rdx',
            hue='Type',
            size='Weights',
            sizes=(10, 200),
            alpha=0.7,
            palette=palette,
            height=height)
        plot.ax.axis('off')
        if title is not None:
            plot.set(title=title)

    return plot


def pairplot(X,
             labels=None,
             col_names=None,
             height=2.5,
             title=None,
             context='talk',
             font_scale=1,
             xticklabels=False,
             yticklabels=False):
    """
    TODO: Update docstring
    """
    if col_names is None:
        col_names = [
            'Dimension {}'.format(i) for i in range(1, X.shape[1] + 1)
        ]

    if labels is not None:
        df_labels = pd.DataFrame(labels, columns=['labels'])
        df = pd.concat([df_labels, df], axis=1)
        df = pd.DataFrame(X, columns=col_names)
    else:
        df = pd.DataFrame(X, columns=col_names)

    if not xticklabels:
        xticklabels = []
        xticks = []
    if not yticklabels:
        yticklabels = []
        yticks = []

    with sns.plotting_context(context, font_scale=font_scale):
        if labels is not None:
            pairs = sns.pairplot(
                df, hue='labels', vars=col_names[:10], height=height)
        else:
            pairs = sns.pairplot(df, vars=col_names[:10], height=height)

        pairs.set(
            xticklabels=xticklabels,
            xticks=xticks,
            yticklabels=yticklabels,
            yticks=yticks)

    return pairs