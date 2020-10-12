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


def _check_data(data):
    if not isinstance(data, np.ndarray):
        raise TypeError("Data much be a np.ndarray.")
    if data.ndim != 2:
        raise ValueError("Data must have dimension 2.")


def _check_item_in_meta(meta, item, name):
    if item is None:
        return []
    if isinstance(item, str):
        item = [item]
    else:
        try:
            iter(item)
        except TypeError:
            msg = (
                f"{name} must be an iterable or string corresponding to columns in meta"
            )
            raise TypeError(msg)
    for col_name in item:
        if (col_name not in meta.columns):
            if (name == "class_order") and (col_name == "size"):
                pass
            else:
                raise ValueError(f"{col_name} is not a column in the meta dataframe.")
    if (name == "class_order"):
        item = item[0]

    return item


def _check_length(item, name, length):
    if length != len(item):
        raise ValueError(
            f"Length of {name} must be the same as corresponding data axis"
        )


def _check_sorting_kws(length, meta, group_class, class_order, item_order, color_class):
    if isinstance(meta, pd.DataFrame):
        # if meta is here, than everything else must be column item in meta
        _check_length(meta, "meta", length)
        group_class = _check_item_in_meta(meta, group_class, "group_class")
        class_order = _check_item_in_meta(meta, class_order, "class_order")
        item_order = _check_item_in_meta(meta, item_order, "item_order")
        colors = _check_item_in_meta(meta, color_class, "color_class")
    else:
        # otherwise, arguments can be a hodgepodge of stuff
        group_class_meta, group_class = _item_to_df(group_class, "group_class", length)
        class_order_meta, class_order = _item_to_df(class_order, "class_order", length)
        item_order_meta, item_order = _item_to_df(item_order, "item_order", length)
        color_class_meta, color_class = _item_to_df(color_class, "color_class", length)
        metas = []
        for m in [group_class_meta, class_order_meta, item_order_meta, color_class_meta]:
            if m is not None:
                metas.append(m)
        if len(metas) > 0:
            meta = pd.concat(metas, axis=1)
        else:
            meta = pd.DataFrame()

    return meta, group_class, class_order, item_order, colors


def _get_separator_info(meta, group_class):
    if meta is None or group_class is None:
        return None

    front = meta.groupby(by=group_class, sort=False).first()
    front_sep_inds = front["sort_idx"].values
    end = meta.groupby(by=group_class, sort=False).last()
    back_sep_inds = end["sort_idx"].values

    return front_sep_inds, back_sep_inds


def _item_to_df(item, name, length):
    if item is None:
        return None, []

    if isinstance(item, pd.Series):
        _check_length(item, name, length)
        item_meta = item.to_frame(name=f"{name}_0")
    elif isinstance(item, list):
        if len(item) == length:  # assuming elements of list are metadata
            item = [item]
        for elem in item:
            _check_length(elem, name, length)
        item_meta = pd.DataFrame({f"{name}_{i}": elem for i, elem in enumerate(item)})
    elif isinstance(item, np.ndarray):
        if item.ndim > 2:
            raise ValueError(f"Numpy array passed as {name} must be 1 or 2d.")
        _check_length(item, name, length)
        if item.ndim < 2:
            item = np.atleast_2d(item).T
        item_meta = pd.DataFrame(
            data=item, columns=[f"{name}_{i}" for i in range(item.shape[1])]
        )
    else:
        raise ValueError(f"{name} must be a pd.Series, np.array, or list.")

    item = list(item_meta.columns.values)

    return item_meta, item


def _remove_shared_ax(ax):
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


def draw_colors(ax, ax_type="x", meta=None, divider=None, color_class=None, palette="tab10"):
    """
    Draw colormap onto the axis to separate the data

    Parameters
    ----------
    ax : matplotlib axes object
        Axes in which to draw the colormap
    ax_type : char, optional
        Setting either the x or y axis, by default "x"
    meta : pd.DataFrame, pd.Series, list of pd.Series or np.array, optional
        Metadata of the matrix such as class, cell type, etc., by default None
    divider : AxesLocator, optional
        Divider used to add new axes to the plot
    color_class : dict, optional
        Attribute in meta by which to draw colorbars, by default None
    palette : str, optional
        Colormap of the colorbar, by default "tab10"

    Returns
    -------
    ax : matplotlib axes object
        Axes in which to draw the color map
    """
    classes = meta[color_class]
    uni_classes = np.unique(classes)
    # Create the color dictionary
    if isinstance(palette, dict):
        color_dict = palette
    elif isinstance(palette, str):
        color_dict = dict(zip(uni_classes, sns.color_palette(palette, len(uni_classes))))

    # Make the colormap
    class_map = dict(zip(uni_classes, range(len(uni_classes))))
    color_sorted = np.vectorize(color_dict.get)(uni_classes)
    color_sorted = np.array(color_sorted)
    if len(color_sorted) != len(uni_classes):
        color_sorted = color_sorted.T
    lc = ListedColormap(color_sorted)
    class_indicator = np.vectorize(class_map.get)(classes)

    if ax_type == "x":
        class_indicator = class_indicator.reshape(1, len(classes))
    elif ax_type == "y":
        class_indicator = class_indicator.reshape(len(classes), 1)
    sns.heatmap(
        class_indicator,
        cmap=lc,
        cbar=False,
        yticklabels=False,
        xticklabels=False,
        ax=ax,
        square=False,
    )
    if ax_type == "x":
        ax.set_xlabel(color_class)
        ax.xaxis.set_label_position('top')
    elif ax_type == "y":
        ax.set_ylabel(color_class)
    return ax


def draw_separators(ax, ax_type="x", meta=None, group_class=None, plot_type="heatmap", gridline_kws=None):
    """
    Draw separators between groups on the plot

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
    plot_type : str, optional
        One of "heatmap" or "scattermap", by default "heatmap"
    gridline_kws : dict, optional
        Plotting arguments for the separators, by default None

    Returns
    -------
    ax : matplotlib axes object
        Axes in which to draw the ticks
    """
    if len(group_class) > 0:
        if gridline_kws is None:
            gridline_kws = dict(color="grey", linestyle="--", alpha=0.7, linewidth=1)

        front_sep_inds, back_sep_inds = _get_separator_info(meta, group_class)
        if plot_type == "heatmap":
            back_sep_inds = back_sep_inds + 1
        if plot_type == "scattermap":
            front_sep_inds = front_sep_inds - 0.5
            back_sep_inds = back_sep_inds + 0.5

        if ax_type == "x":
            lims = ax.get_xlim()
            drawer = ax.axvline
        elif ax_type == "y":
            lims = ax.get_ylim()
            drawer = ax.axhline

        # Draw the separators
        for t in list(front_sep_inds):
            if t not in lims:
                drawer(t, **gridline_kws)
        for t in list(back_sep_inds):
            if t not in lims:
                drawer(t, **gridline_kws)

    return ax


def draw_ticks(ax, ax_type="x", meta=None, group_class=None, group_border=True, plot_type="heatmap"):
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
    group_border : bool, optional
        Whether to draw separator on the tick axes, by default True
    plot_type : str, optional
        One of "heatmap" or "scattermap", by default "heatmap"

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
        # Identify the center of each class
        center = meta.groupby(by=group_class, sort=False).mean()
        tick_inds = np.array(center["sort_idx"].values)
        if plot_type == "heatmap":
            tick_inds = tick_inds + 0.5
        tick_labels = list(center.index.get_level_values(group_class[0]))

    if ax_type == "x":
        ax.set_xticks(tick_inds)
        ax.set_xticklabels(tick_labels, ha="center", va="bottom")
        ax.xaxis.tick_top()
        ax.set_xlabel(group_class[0])
        ax.xaxis.set_label_position('top') 
    elif ax_type == "y":
        ax.set_yticks(tick_inds)
        ax.set_yticklabels(tick_labels, ha="right", va="center")
        ax.set_ylabel(group_class[0])

    if group_border:
        front_sep_inds, back_sep_inds = _get_separator_info(meta, group_class)
        if plot_type == "heatmap":
            back_sep_inds = back_sep_inds + 1
        if plot_type == "scattermap":
            front_sep_inds = front_sep_inds - 0.5
            back_sep_inds = back_sep_inds + 0.5

        if ax_type == "x":
            drawer = ax.axvline
        elif ax_type == "y":
            drawer = ax.axhline

        for t in list(front_sep_inds):
            drawer(t, color="black", linestyle="--", alpha=0.7, linewidth=1)
        for t in list(back_sep_inds):
            drawer(t, color="black", linestyle="--", alpha=0.7, linewidth=1)

    return ax


def gridmap(data, ax=None, legend=False, sizes=(5, 10), **kws):
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
    # ax.axis("image")
    ax.set_xlim((-1, n_verts + 1))
    ax.set_ylim((n_verts + 1, -1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    # ax.axis("off")
    # plt.show()
    return ax


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
    meta["sort_idx"] = range(len(meta))
    return inds, meta


def matrixplot(
    data,
    ax=None,
    meta=None,
    plot_type="heatmap",
    group_class=None,
    class_order="size",
    item_order=None,
    color_class=None,
    highlight=None,
    palette="tab10",
    ticks=True,
    tick_pad=None,
    color_pad=None,
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
    color_class : dict, optional
        Attribute in meta by which to draw colorbars, by default None
    border : bool, optional
        Whether the plot should have border, by default True
    cmap : str, optional
        Colormap of the heatmap, by default "RdBu_r"
    palette : str, optional
        Colormap of the colorbar, by default "tab10"
    ticks : bool, optional
        Whether the plot has ticks, by default True
    tick_pad : int, float, optional
        Custom padding to use for the distance between tick axes, by default None
    color_pad : int, float, optional
        Custom padding to use for the distance between color axes, by default None
    center : int, optional
        The value at which to center the colormap when plotting divergant data., by default 0
    sizes : tuple, optional
        Min and max sizes of dots, by default (5, 10)
    square : bool, optional
        Whether the plot should be square, by default False
    gridline_kws : dict, optional
        Plotting arguments for the separators, by default None
    spinestyle_kws : dict, optional
        Plotting arguments for the spine border, by default None
    highlight_kws : dict, optional
        Plotting arguments for the highlighted separators, by default None
    **kwargs : dict, optional
        Additional plotting arguments

    Returns
    -------
    ax : matplotlib axes object
        Axes in which to draw the plot, by default None
    divider : AxesLocator
        Divider used to add new axes to the plot
    """
    # check for the type and dimension of the data
    _check_data(data)

    # check for the plot type
    plot_type_opts = ["scattermap", "heatmap"]
    if plot_type not in plot_type_opts:
        raise ValueError(f"`plot_type` must be one of {plot_type_opts}")

    # check for the types of the sorting arguments
    meta, group_class, class_order, item_order, color_class = _check_sorting_kws(
        data.shape[0],
        meta,
        group_class,
        class_order,
        item_order,
        color_class,
    )

    # assign the sorting method to the row and column
    row_meta = meta
    row_group_class = group_class
    row_class_order = class_order
    row_item_order = item_order
    row_color_class = color_class
    row_palette = palette
    col_meta = meta
    col_group_class = group_class
    col_class_order = class_order
    col_item_order = item_order
    col_color_class = color_class
    col_palette = palette

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
        sns.heatmap(data, cmap=cmap, ax=ax, center=center, xticklabels=False, yticklabels=False, **kws)
    elif plot_type == "scattermap":
        gridmap(data, ax=ax, sizes=sizes, **kws)

    # extra features of the graph
    if square:
        ax.axis("square")
    if plot_type == "scattermap":
        ax_pad = 0.5
    else:
        ax_pad = 0
    ax.set_ylim(data.shape[0] - ax_pad, 0 - ax_pad)
    ax.set_xlim(0 - ax_pad, data.shape[1] - ax_pad)
    # plt.show()

    # ticks, separators, colors, and spine not implemented yet
    # make_axes_locatable allows us to create new axes like colormap
    divider = make_axes_locatable(ax)

    # draw separators
    draw_separators(
        ax,
        ax_type="x",
        meta=col_meta,
        group_class=col_group_class,
        plot_type=plot_type,
        gridline_kws=gridline_kws,
    )
    draw_separators(
        ax,
        ax_type="y",
        meta=row_meta,
        group_class=row_group_class,
        plot_type=plot_type,
        gridline_kws=gridline_kws,
    )
    # plt.show()

    # draw highlighted separators
    if highlight_kws is None:
        highlight_kws = dict(color="black", linestyle="-", linewidth=1)
    if highlight is not None:
        draw_separators(
            ax,
            ax_type="x",
            meta=col_meta,
            group_class=highlight,
            plot_type=plot_type,
            gridline_kws=highlight_kws,
        )
        draw_separators(
            ax,
            ax_type="y",
            meta=col_meta,
            group_class=highlight,
            plot_type=plot_type,
            gridline_kws=highlight_kws,
        )

    first_ax = True
    # draw colors
    if len(color_class) > 0:
        if color_pad is None:
            color_pad = len(color_class) * [0.5]
        if first_ax:
            color_pad[0] = 0
            first_ax = False

        rev_color_class = list(color_class[::-1])
        for i, sc in enumerate(rev_color_class):
            color_ax = divider.append_axes("top", size="3%", pad=color_pad[i], sharex=ax)
            _remove_shared_ax(color_ax)
            draw_colors(
                color_ax,
                meta=col_meta,
                divider=divider,
                ax_type="x",
                color_class=rev_color_class[i],
                palette=col_palette,
            )
            # plt.show()
            color_ax = divider.append_axes("left", size="3%", pad=color_pad[i], sharey=ax)
            _remove_shared_ax(color_ax)
            draw_colors(
                color_ax,
                meta=row_meta,
                divider=divider,
                ax_type="y",
                color_class=rev_color_class[i],
                palette=row_palette,
            )
            # plt.show()
    # plt.show()

    # draw ticks
    if len(group_class) > 0 and ticks:
        if tick_pad is None:
            tick_pad = len(group_class) * [0.5]
        if first_ax:
            tick_pad[0] = 0
            first_ax = False

        # Reverse the order of the group class, so the ticks are drawn in the opposite order
        rev_group_class = list(group_class[::-1])
        for i, sc in enumerate(rev_group_class):

            # Add a new axis when needed
            tick_ax = divider.append_axes("top", size="1%", pad=tick_pad[i], sharex=ax)
            _remove_shared_ax(tick_ax)
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
                plot_type=plot_type,
            )
            ax.xaxis.set_label_position("top")

            # Add a new axis when needed
            tick_ax = divider.append_axes("left", size="1%", pad=tick_pad[i], sharey=ax)
            _remove_shared_ax(tick_ax)
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
                plot_type=plot_type,
            )

    # spines
    if spinestyle_kws is None:
        spinestyle_kws = dict(linestyle="-", linewidth=1, alpha=0.7)
    if border:
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(spinestyle_kws["linewidth"])
            spine.set_linestyle(spinestyle_kws["linestyle"])
            spine.set_alpha(spinestyle_kws["alpha"])

    return ax, divider


def main():
    N = 10
    data = np.random.randint(10, size=(N, N))
    meta = pd.DataFrame({
        'hemisphere': np.random.randint(3, size=N),
        'dVNC': np.random.randint(3, size=N),
        'ID': np.random.randint(10, size=N)
    })
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    matrixplot(
        data=data,
        ax=ax,
        meta=meta,
        plot_type="heatmap",
        group_class=["hemisphere", "dVNC"],
        item_order=["ID"],
        sizes=(1, 5),
        color_class=["hemisphere", "dVNC"],
        highlight=["hemisphere"]
    )
    plt.show()


if __name__ == "__main__":
    main()

