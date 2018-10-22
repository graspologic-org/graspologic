# omni.py
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


def plot_heatmap(X,
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
    Parameters
    ----------
    X : nx.Graph or np.ndarray object
        Graph or numpy matrix to plot
    transform : None, or one of {log, pass_to_ranks}
    figsize : tuple of integers, optional, default: (10, 10)
        Width, height in inches.
    title : str, optional, default: None
        Title of plot.
    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font elements.
    xticklabels, yticklabels : bool or list, optional
        If True, plot the column names of the dataframe. If False, don’t plot the 
        column names. If list-like, plot these alternate labels as the xticklabels.
        If an integer, use the column names but plot only every n label. 
    cmap : str
        Valid color map.
    center : float, optional, default: None
        The value at which to center the colormap when plotting divergant data.
    cbar : bool, default: True
        Whether to draw a colorbar.
    """
    arr = import_graph(X)

    if transform == 'log':
        #arr = np.log(arr, where=(arr > 0))
        #hacky, but np.log(arr, where=arr>0) is really buggy
        arr = arr.copy()
        arr[arr > 0] = np.log(arr[arr > 0])
    elif transform == 'pass_to_ranks':
        arr = pass_to_ranks(arr)

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


def plot_grid_plot(X,
                   labels,
                   transform=None,
                   height=10,
                   title=None,
                   context='talk',
                   font_scale=1,
                   xticklabels=False,
                   yticklabels=False):
    """
    Parameters
    ----------
    X : list of nx.Graph or np.ndarray object
        List of nx.Graph or numpy arrays to plot
    transform : None, or one of {log, pass_to_ranks}
    height : integers, optional, default: 10
        Height of figure in inches.
    title : str, optional, default: None
        Title of plot.
    context :  None, or one of {paper, notebook, talk (default), poster}
        The name of a preconfigured set.
    font_scale : float, optional, default: 1
        Separate scaling factor to independently scale the size of the font elements.
    xticklabels, yticklabels : bool or list, optional
        If True, plot the column names of the dataframe. If False, don’t plot the 
        column names. If list-like, plot these alternate labels as the xticklabels.
        If an integer, use the column names but plot only every n label. 
    """

    if isinstance(X, list):
        graphs = [import_graph(x) for x in X]
    else:
        graphs = [import_graph(X)]

    # TODO: add transforms

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