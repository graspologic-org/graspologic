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


def _get_separator_info(meta, group_class):
    if meta is None or group_class is None:
        return None

    front = meta.groupby(by=group_class, sort=False).first()
    sep_inds = list(front["sort_idx"].values - 0.5)
    end = meta.groupby(by=group_class, sort=False).last()
    sep_inds.extend(list(end["sort_idx"].values + 0.5))

    return sep_inds


def draw_ticks(ax, ax_type="x", meta=None, group_class=None, tick_rot=0, group_border=True):
    """
    Draw ticks onto the axis of the plot to separate the data

    Parameters
    ----------
    ax : matplotlib axes object
        Axes in which to draw the ticks
    ax_type : char, optional
        Setting either the x or y axis, by default "x"
    meta : pd.DataFrame, pd.Series, list of pd.Series or np.array, optional
        Metadata of the matrix such as class, cell type, etc., by default None
    group_class : list or np.ndarray, optional
        Metadata to group the graph in the plot, by default None
    tick_rot : int, optional
        [description], by default 0
    group_border : bool, optional
        [description], by default True

    Returns
    -------
    ax : matplotlib axes object
        Axes in which to draw the ticks
    """
    # Identify the locations of the ticks
    if meta is None or group_class is None:
        tick_inds = None
        tick_labels = None
    else:
        meta["sort_idx"] = range(len(meta))
        # Identify the center of each class
        center = meta.groupby(by=group_class, sort=False).mean()
        tick_inds = np.array(center["sort_idx"].values)
        tick_labels = list(center.index.get_level_values(group_class[0]))

    if ax_type == "x":
        ax.set_xticks(tick_inds)
        ax.set_xticklabels(tick_labels, rotation=tick_rot, ha="center", va="bottom")
        ax.xaxis.tick_top()
        ax.set_xlabel(group_class[0])
        ax.xaxis.set_label_position('top') 
    elif ax_type == "y":
        ax.set_yticks(tick_inds)
        ax.set_yticklabels(tick_labels, ha="right", va="center")
        ax.set_ylabel(group_class[0])

    if group_border:
        sep_inds = _get_separator_info(meta, group_class)
        for t in sep_inds:
            if ax_type == "x":
                ax.axvline(t, color="black", linestyle="--", alpha=0.7, linewidth=1)
            elif ax_type == "y":
                ax.axhline(t, color="black", linestyle="--", alpha=0.7, linewidth=1)

    return ax


def gridmap(data, ax=None, legend=False, sizes=(5, 10), spines=False, border=True, **kws):
    """
    Draw a scattermap of the data with extra grid features

    Parameters
    ----------
    data : np.narray, ndim=2
        Matrix to plot
    ax: matplotlib axes object, optional
        Axes in which to draw the plot, by default None
    legend : bool, optional
        [description], by default False
    sizes : tuple, optional
        min and max of dot sizes, by default (5, 10)
    spines : bool, optional
        whether to keep the spines of the plot, by default False
    border : bool, optional
        [description], by default True
    **kws : dict, optional
        Additional plotting arguments

    Returns
    -------
    ax: matplotlib axes object, optional
        Axes in which to draw the plot, by default None
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
    sns.scatterplot(
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
    # plt.show()
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
    ax.set_ylabel("")
    ax.set_xlabel("")
    # ax.axis("off")
    return ax


def remove_shared_ax(ax):
    """
    Remove ax from its sharex and sharey
    """
    # Remove ax from the Grouper object
    shax = ax.get_shared_x_axes()
    shay = ax.get_shared_y_axes()
    shax.remove(ax)
    shay.remove(ax)

    # Set a new ticker with the respective new locator and formatter
    for axis in [ax.xaxis, ax.yaxis]:
        ticker = mpl.axis.Ticker()
        axis.major = ticker
        axis.minor = ticker
        # No ticks and no labels
        loc = mpl.ticker.NullLocator()
        fmt = mpl.ticker.NullFormatter()
        axis.set_major_locator(loc)
        axis.set_major_formatter(fmt)
        axis.set_minor_locator(loc)
        axis.set_minor_formatter(fmt)


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
    length : int
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

    # TODO sort each group based on the class size in each group rather than
    # the size of the entire class
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
    tick_rot=0,
    tick_pad=None,
    border=True,
    center=0,
    cmap="RdBu_r",
    sizes=(5, 10),
    square=True,
    gridline_kws=None,
    spinestyle_kws=None,
    highlight_kws=None,
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
        whether the plot has ticks, by default True
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
        gridmap(data, ax=ax, sizes=sizes, border=False, **kws)

    # extra features of the graph
    if square:
        ax.axis("square")
    if plot_type == "scattermap":
        ax_pad = 0.5
    else:
        ax_pad = 0
    ax.set_ylim(data.shape[0] + ax_pad, 0 - ax_pad)
    ax.set_xlim(0 - ax_pad, data.shape[1] + ax_pad)

    # ticks, separators, colors, and spine not implemented yet
    # make_axes_locatable allows us to create new axes like colormap
    divider = make_axes_locatable(ax)

    top_ax = ax
    left_ax = ax

    # draw ticks
    if len(group_class) > 0 and ticks:
        if tick_pad is None:
            tick_pad = len(group_class) * [0.5]

        # Reverse the order of the group class, so the ticks are drawn in the opposite order
        rev_group_class = list(group_class[::-1])
        for i, sc in enumerate(rev_group_class):

            tick_ax = top_ax
            # Add a new axis when needed
            if i > 0:
                tick_ax = divider.append_axes("top", size="1%", pad=tick_pad[i], sharex=ax)
                remove_shared_ax(tick_ax)
                tick_ax.spines["right"].set_visible(False)
                tick_ax.spines["top"].set_visible(True)
                tick_ax.spines["left"].set_visible(False)
                tick_ax.spines["bottom"].set_visible(False)

            # Draw the ticks for the x axis
            draw_ticks(
                tick_ax,
                ax_type="x",
                meta=col_meta,
                group_class=rev_group_class[i:],
            )
            ax.xaxis.set_label_position("top")

            tick_ax = left_ax
            # Add a new axis when needed
            if i > 0:
                tick_ax = divider.append_axes("left", size="1%", pad=tick_pad[i], sharey=ax)
                remove_shared_ax(tick_ax)
                tick_ax.spines["right"].set_visible(False)
                tick_ax.spines["top"].set_visible(False)
                tick_ax.spines["left"].set_visible(True)
                tick_ax.spines["bottom"].set_visible(False)

            # Draw the ticks for the y axis
            draw_ticks(
                tick_ax,
                ax_type="y",
                meta=row_meta,
                group_class=rev_group_class[i:],
            )

    # draw separators

    # draw colors

    # spines
    if spinestyle_kws is None:
        spinestyle_kws = dict(linestyle="-", linewidth=1, alpha=0.7)
    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(spinestyle_kws["linewidth"])
            spine.set_linestyle(spinestyle_kws["linestyle"])
            spine.set_alpha(spinestyle_kws["alpha"])

    return ax, divider, top_ax, left_ax


def main():
    N = 50
    data = np.random.randint(10, size=(N, N))
    meta = pd.DataFrame({
        'hemisphere': np.random.randint(2, size=N),
        'dVNC': np.random.randint(2, size=N),
        'ID': np.random.randint(10, size=N)
    })
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    matrixplot(
        data=data,
        ax=ax,
        meta=meta,
        plot_type="scattermap",
        group_class=["hemisphere", "dVNC"],
        item_order=["ID"],
        sizes=(1, 5)
    )
    plt.show()


if __name__ == "__main__":
    main()

