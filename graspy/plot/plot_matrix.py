import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib as mpl
from matplotlib.colors import ListedColormap


"""
This file is adopted from both graspy/plot/plot.py
and maggot_models/src/visualization/matrix.py.
It plots a data matrix sorted by some metadata,
it also provides some visualization map like color map and ticks
"""


def gridmap(data, ax=None, legend=False, sizes=(5, 10), spines=False, border=True, **kws):
    """
    Draw a scattermap of the data with extra grid features

    Parameters
    ----------
    data: np.narray, ndim=2
        Matrix to plot
    ax : matplotlib axes object, optional
        Axes in which to draw the plot, by default None
    legend: bool, optional
        [description], by default False
    sizes: tuple, optional
        [description], by default (5, 10)
    spines: bool, optional
        whether to keep the spines of the plot, by default False
    border: bool, optional
        [description], by default True
    **kws: dict, optional
        Additional plotting arguments

    Returns
    -------
    [type]
        [description]
    """

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(20, 20))
    n_verts = data.shape[0]
    inds = np.nonzero(data)
    edges = data[inds]
    scatter_df = pd.DataFrame()
    scatter_df["weight"] = edges
    scatter_df["x"] = inds[1]
    scatter_df["y"] = inds[0]
    ax = sns.scatterplot(
        data=scatter_df,
        x="x",
        y="y",
        size="weight",
        legend=legend,
        sizes=sizes,
        ax=ax,
        linewidth=0,
        **kws,
    )
    plt.show()
    # ax.axis("image")
    ax.set_xlim((-1, n_verts + 1))
    ax.set_ylim((n_verts + 1, -1))
    if not spines:
        remove_spines(ax)
    if border:
        linestyle_kws = {
            "linestyle": "--",
            "alpha": 0.5,
            "linewidth": 0.5,
            "color": "grey",
        }
        ax.axvline(0, **linestyle_kws)
        ax.axvline(n_verts, **linestyle_kws)
        ax.axhline(0, **linestyle_kws)
        ax.axhline(n_verts, **linestyle_kws)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("Vertices")
    ax.set_xlabel("Vertices")
    # ax.axis("off")
    return ax


def remove_spines(ax, keep_corner=False):
    """
    Removes the lines noting the data area boundaries

    Parameters
    ----------
    ax : matplotlib axes object
        Axes in which to draw the plot
    keep_corner : bool, optional
        whether to keep the bottom left corner, by default False
    """

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if not keep_corner:
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)


def sort_meta(length, meta, group_class, class_order="size", item_order=None):
    """
    Sort the data and metadata according to the sorting method

    Parameters
    ----------
    length: int
        Number of nodes
    meta : pd.DataFrame, pd.Series, list of pd.Series or np.array, optional
        Metadata of the matrix such as class, cell type, etc., by default None
    group_class : list or np.ndarray, optional
        Metadata to group the graph in the plot, by default None
    class_order : str, optional
        Attribute of the sorting class to sort classes within the graph, by default "size"
    item_order : string or list of string, optional
        Attribute in meta by which to sort elements within a class, by default None

    Returns
    -------
    inds :
        The index of the data after sorting
    meta :
        The metadata after sorting
    """

    # if metadata does not exist, return the original data
    if meta is None or len(meta) == 0:
        return np.arange(length), meta

    meta = meta.copy()
    # total_sort_by keeps track of what attribute was used to sort
    total_sort_by = []
    # create new columns in the dataframe that correspond to the sorting order
    # one problem with this current sorting algorithm is that we cannot sort
    # classes by size and other class attributes at the same time.
    for gc in group_class:
        if class_order == "size":
            class_size = meta.groupby(gc).size()
            # map each node from the sorting class to their sorting class size
            meta[f"{gc}_size_order"] = meta[gc].map(class_size)
            total_sort_by.append(f"{gc}_size_order")
        elif len(class_order) > 0:
            for co in class_order:
                class_value = meta.groupby(gc)[co].mean()
                # map each node from the sorting class to certain sorting class attribute
                meta[f"{gc}_{co}_order"] = meta[sc].map(class_value)
                total_sort_by.append(f"{gc}_{co}_order")
        total_sort_by.append(gc)
    total_sort_by += item_order

    # arbitrarily sort the data from 1 to length
    meta["sort_idx"] = range(len(meta))
    # if we actually need to sort, sort by class_order, sort_class, item_order
    if len(total_sort_by) > 0:
        meta.sort_values(total_sort_by, inplace=True, kind="mergesort")

    inds = meta["sort_idx"].values
    return inds, meta


def matrixplot(
    data,
    ax=None,
    meta=None,
    plot_type="heatmap",
    group_class=None,
    class_order="size",
    item_order=None,
    colors=None,
    highlight=None,
    palette="tab10",
    ticks=True,
    border=True,
    tick_rot=0,
    center=0,
    cmap="RdBu_r",
    sizes=(5, 10),
    square=True,
    gridline_kws=None,
    spinestyle_kws=None,
    highlight_kws=None,
    col_tick_pad=None,
    row_tick_pad=None,
    **kws
):
    """
    Sorts a matrix and plots it with ticks and colors on the borders

    Parameters
    ----------
    data : np.ndarray, ndim=2
        Matrix to plot
    ax : matplotlib axes object, optional
        Axes in which to draw the plot, by default None
    plot_type : str, optional
        One of "heatmap" or "scattermap", by default "heatmap"
    meta : pd.DataFrame, pd.Series, list of pd.Series or np.array, optional
        Metadata of the matrix such as class, cell type, etc., by default None
    group_class : list or np.ndarray, optional
        Metadata to group the graph in the plot, by default None
    class_order : str, optional
        Attribute of the sorting class to sort classes within the graph, by default "size"
    item_order : string or list of string, optional
        Attribute in meta by which to sort elements within a class, by default None
    border : bool, optional
        [description], by default True
    cmap : str, optional
        [description], by default "RdBu_r"
    colors : dict, optional
        [description], by default None
    palette : str, optional
        [description], by default "tab10"
    ticks : bool, optional
        [description], by default True
    tick_rot : int, optional
        [description], by default 0
    tick_pad : int, float, optional
        Custom padding to use for the distance between ticks, by default None
    center : int, optional
        [description], by default 0
    sizes : tuple, optional
        min and max sizes of dots, by default (5, 10)
    square : bool, optional
        [description], by default False
    gridline_kws : [type], optional
        [description], by default None
    spinestyle_kws : [type], optional
        [description], by default None
    highlight_kws : [type], optional
        [description], by default None
    **kwargs : dict, optional
        Additional plotting arguments

    Returns
    -------
    [type]
        [description]
    ax : [type]
        [description]
    divider : [type]
        [description]
    top_cax : [type]
        [description]
    left_cax : [type]
        [description]
    """
    # check for input has not been implemented yet

    # assign the sorting method to the row and column
    row_meta = meta
    row_group_class = group_class
    row_class_order = class_order
    row_item_order = item_order
    row_colors = colors
    col_meta = meta
    col_group_class = group_class
    col_class_order = class_order
    col_item_order = item_order
    col_colors = colors   

    # sort the data and the metadata according to the sorting method
    row_inds, row_meta = sort_meta(
        data.shape[0],
        row_meta,
        row_group_class,
        class_order=row_class_order,
        item_order=row_item_order,
    )
    col_inds, col_meta = sort_meta(
        data.shape[1],
        col_meta,
        col_group_class,
        class_order=col_class_order,
        item_order=col_item_order,
    )
    data = data[np.ix_(row_inds, col_inds)]

    # draw the main heatmap/scattermap
    if ax is None:
        _, ax = plt.subplot(1, 1, figsize=(10, 10))

    if plot_type == "heatmap":
        sns.heatmap(data, cmap=cmap, ax=ax, center=center, **kws)
    elif plot_type == "scattermap":
        ax = gridmap(data, ax=ax, sizes=sizes, border=False, **kws)

    # extra features of the graph
    if square:
        ax.axis("square")
    if plot_type == "scattermap":
        ax_pad = 0.5
    else:
        ax_pad = 0
    ax.set_ylim(data.shape[0] + ax_pad, 0 - ax_pad)
    ax.set_xlim(0 - ax_pad, data.shape[1] + ax_pad)

    # ticks, separator, colors, and spine not implemented yet
    # make_axes_locatable allows us to create new axes like colormap
    divider = make_axes_locatable(ax)

    # spines
    if spinestyle_kws is None:
        spinestyle_kws = dict(linestyle="-", linewidth=1, alpha=0.7)
    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(spinestyle_kws["linewidth"])
            spine.set_linestyle(spinestyle_kws["linestyle"])
            spine.set_alpha(spinestyle_kws["alpha"])

    return ax


def main():

    data = np.random.randint(10, size=(50, 50))
    meta = pd.DataFrame({
        'hemisphere': np.random.randint(2, size=50),
        'ID': np.random.randint(10, size=50)
    })
    fig, ax = plt.subplots(1, 1, figsize=(15, 15))
    matrixplot(
        data=data,
        ax=ax,
        meta=meta,
        plot_type="scattermap",
        group_class=["hemisphere"],
        item_order=["ID"],
        sizes=(1, 5)
    )


if __name__ == "__main__":
    main()

