# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

# type: ignore

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import csr_matrix


def _check_data(data, plot_type):
    if plot_type == "scattermap":
        if not isinstance(data, (np.ndarray, csr_matrix)):
            raise TypeError("`data` must be a np.ndarray or scipy.sparse.csr_matrix.")
    elif plot_type == "heatmap":
        if not isinstance(data, np.ndarray):
            msg = "`data` must be a np.ndarray. If your data is a sparse matrix, please"
            msg += " make sure that `plot_type` is set to 'scattermap'"
            raise TypeError(msg)
    if data.ndim != 2:
        raise ValueError("`data` must have dimension 2.")


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
        if col_name not in meta.columns:
            if (name == "group_order") and (col_name == "size"):
                pass
            else:
                raise ValueError(f"{col_name} is not a column in the meta dataframe.")

    return item


def _check_length(item, name, length):
    if length != len(item):
        raise ValueError(
            f"Length of {name} must be the same as corresponding data axis"
        )


def _check_plotting_kws(palette, color):
    if isinstance(palette, str) or isinstance(palette, dict):
        palette = [palette]
    elif len(palette) != len(color) and len(palette) != 1:
        raise ValueError("There is not enough palette value for each color class")

    return palette


def _check_sorting_kws(length, meta, group, group_order, item_order, color, highlight):
    if isinstance(meta, pd.DataFrame):
        # if meta is here, than everything else must be column item in meta
        _check_length(meta, "meta", length)
        group = _check_item_in_meta(meta, group, "group")
        group_order = _check_item_in_meta(meta, group_order, "group_order")
        item_order = _check_item_in_meta(meta, item_order, "item_order")
        color = _check_item_in_meta(meta, color, "color")
        highlight = _check_item_in_meta(meta, highlight, "highlight")
    else:
        # otherwise, arguments can be a hodgepodge of stuff
        group_meta, group = _item_to_df(group, "group", length)
        group_order_meta, group_order = _item_to_df(group_order, "group_order", length)
        item_order_meta, item_order = _item_to_df(item_order, "item_order", length)
        color_meta, color = _item_to_df(color, "color", length)
        highlight_meta, highlight = _item_to_df(highlight, "highlight", length)
        metas = []
        for m in [
            group_meta,
            group_order_meta,
            item_order_meta,
            color_meta,
            highlight_meta,
        ]:
            if m is not None:
                metas.append(m)
        if len(metas) > 0:
            meta = pd.concat(metas, axis=1)
        else:
            meta = pd.DataFrame()

    return meta, group, group_order, item_order, color, highlight


def _get_separator_info(meta, group):
    if meta is None or group is None:
        return None

    sep = meta.groupby(by=group, sort=False).first()
    sep_inds = sep["sort_idx"].values
    sep_inds = list(sep["sort_idx"].values)
    last = meta.groupby(group, sort=False).last()
    sep_inds.append(last["sort_idx"].values[-1] + 1)
    return np.array(sep_inds)


def _item_to_df(item, name, length):
    if item is None:
        return None, []
    elif (name == "group_order") and (item == "size"):
        return None, ["size"]

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


def _set_spines(ax, left, right, top, bottom):
    ax.spines["right"].set_visible(right)
    ax.spines["top"].set_visible(top)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)

    return ax


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


def draw_colors(ax, ax_type="x", meta=None, divider=None, color=None, palette="tab10"):
    r"""
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
    color : str, list of str, or array_like, optional
        Attribute in meta by which to draw colorbars, by default None
    palette : str or dict, optional
        Colormap of the colorbar, by default "tab10"

    Returns
    -------
    ax : matplotlib axes object
        Axes in which to draw the color map
    """
    classes = meta[color]
    uni_classes = np.unique(classes)
    # Create the color dictionary
    if isinstance(palette, dict):
        color_dict = palette
    elif isinstance(palette, str):
        color_dict = dict(
            zip(uni_classes, sns.color_palette(palette, len(uni_classes)))
        )

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
        ax.set_xlabel(color, fontsize=20)
        ax.xaxis.set_label_position("top")
    elif ax_type == "y":
        ax.set_ylabel(color, fontsize=20)
    return ax


def draw_separators(
    ax,
    ax_type="x",
    meta=None,
    group=None,
    plot_type="heatmap",
    gridline_kws=None,
):
    r"""
    Draw separators between groups on the plot

    Parameters
    ----------
    ax : matplotlib axes object
        Axes in which to draw the ticks
    ax_type : char, optional
        Setting either the x or y axis, by default "x"
    meta : pd.DataFrame, pd.Series, list of pd.Series or np.array, optional
        Metadata of the matrix such as class, cell type, etc., by default None
    group : str, list of str, or array_like, optional
        Attribute in meta to group the graph in the plot, by default None
    plot_type : str, optional
        One of "heatmap" or "scattermap", by default "heatmap"
    gridline_kws : dict, optional
        Plotting arguments for the separators, by default None

    Returns
    -------
    ax : matplotlib axes object
        Axes in which to draw the ticks
    """
    if len(group) > 0:
        if gridline_kws is None:
            gridline_kws = dict(color="grey", linestyle="--", alpha=0.7, linewidth=1)

        sep_inds = _get_separator_info(meta, group)
        if plot_type == "scattermap":
            sep_inds = sep_inds - 0.5

        if ax_type == "x":
            lims = ax.get_xlim()
            drawer = ax.axvline
        elif ax_type == "y":
            lims = ax.get_ylim()
            drawer = ax.axhline

        # Draw the separators
        for t in list(sep_inds):
            if t not in lims:
                drawer(t, **gridline_kws)

    return ax


def draw_ticks(
    ax,
    ax_type="x",
    meta=None,
    group=None,
    group_border=True,
    plot_type="heatmap",
):
    r"""
    Draw ticks onto the axis of the plot to separate the data

    Parameters
    ----------
    ax : matplotlib axes object
        Axes in which to draw the ticks
    ax_type : char, optional
        Setting either the x or y axis, by default "x"
    meta : pd.DataFrame, pd.Series, list of pd.Series or np.array, optional
        Metadata of the matrix such as class, cell type, etc., by default None
    group : str, list of str, or array_like, optional
        Attribute in meta to group the graph in the plot, by default None
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
    if meta is None or group is None:
        tick_inds = None
        tick_labels = None
    else:
        # Identify the center of each class
        center = meta.groupby(by=group, sort=False).mean()
        tick_inds = np.array(center["sort_idx"].values)
        if plot_type == "heatmap":
            tick_inds = tick_inds + 0.5
        tick_labels = list(center.index.get_level_values(group[0]))

    if ax_type == "x":
        ax.set_xticks(tick_inds)
        ax.set_xticklabels(tick_labels, ha="center", va="bottom", fontsize=20)
        ax.xaxis.tick_top()
        ax.set_xlabel(group[0], fontsize=20)
        ax.xaxis.set_label_position("top")
    elif ax_type == "y":
        ax.set_yticks(tick_inds)
        ax.set_yticklabels(tick_labels, ha="right", va="center", fontsize=20)
        ax.set_ylabel(group[0], fontsize=20)

    if group_border:
        sep_inds = _get_separator_info(meta, group)
        if plot_type == "scattermap":
            sep_inds = sep_inds - 0.5

        if ax_type == "x":
            drawer = ax.axvline
        elif ax_type == "y":
            drawer = ax.axhline

        for t in list(sep_inds):
            drawer(t, color="black", linestyle="-", alpha=1, linewidth=1)

    return ax


def scattermap(data, ax=None, legend=False, sizes=(5, 10), **kws):
    r"""
    Draw a matrix using points instead of a heatmap. Helpful for larger, sparse
    matrices.

    Parameters
    ----------
    data : np.narray, scipy.sparse.csr_matrix, ndim=2
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
    edges = np.squeeze(np.asarray(data[inds]))
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
    ax.set_xlim((-1, n_verts + 1))
    ax.set_ylim((n_verts + 1, -1))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.set_xlabel("")
    return ax


def sort_meta(length, meta, group, group_order=["size"], item_order=None):
    r"""
    Sort the data and metadata according to the sorting method

    Parameters
    ----------
    length : int
        Number of nodes
    meta : pd.DataFrame, pd.Series, list of pd.Series or np.array, optional
        Metadata of the matrix such as class, cell type, etc., by default None
    group : str, list of str, or array_like, optional
        Attribute in meta to group the graph in the plot, by default None
    group_order : str, list of str, or array_like, optional
        Attribute of the sorting class to sort classes within the graph,
        by default "size"
    item_order : str, list of str, or array_like, optional
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

    for gc in group:
        for co in group_order:
            if co == "size":
                class_size = meta.groupby(gc).size()
                # map each node from the sorting class
                # to their sorting class size
                meta[f"{gc}_size_order"] = meta[gc].map(class_size)
                total_sort_by.append(f"{gc}_size_order")
            else:
                class_value = meta.groupby(gc)[co].mean()
                # map each node from the sorting class
                # to certain sorting class attribute
                meta[f"{gc}_{co}_order"] = meta[gc].map(class_value)
                total_sort_by.append(f"{gc}_{co}_order")
        total_sort_by.append(gc)
    total_sort_by += item_order

    # arbitrarily sort the data from 1 to length
    meta["sort_idx"] = range(len(meta))
    # if we actually need to sort, sort by group_order, group, item_order
    if len(total_sort_by) > 0:
        meta.sort_values(total_sort_by, inplace=True, kind="mergesort")

    inds = np.copy(meta["sort_idx"].values)
    meta["sort_idx"] = range(len(meta))
    return inds, meta


def matrixplot(  # type: ignore
    data,
    ax=None,
    col_meta=None,
    row_meta=None,
    plot_type="heatmap",
    col_group=None,
    row_group=None,
    col_group_order="size",
    row_group_order="size",
    col_item_order=None,
    row_item_order=None,
    col_color=None,
    row_color=None,
    col_highlight=None,
    row_highlight=None,
    col_palette="tab10",
    row_palette="tab10",
    col_ticks=True,
    row_ticks=True,
    col_tick_pad=None,
    row_tick_pad=None,
    col_color_pad=None,
    row_color_pad=None,
    border=True,
    center=0,
    cmap="RdBu_r",
    sizes=(5, 10),
    square=True,
    gridline_kws=None,
    spinestyle_kws=None,
    highlight_kws=None,
    **kws,
):
    r"""
    Sorts and plots a matrix in various ways, and with optional information added to the
    margin of the matrix plot.

    Read more in the `Tutorial https://microsoft.github.io/graspologic/tutorials/plotting/matrixplot.html`

    Parameters
    ----------
    data : np.ndarray or scipy.sparse.csr_matrix with ndim=2
        Matrix to plot. Sparse matrix input is only accepted if ``plot_type == 'scattermap'``.
    ax : matplotlib axes object (default=None)
        Axes in which to draw the plot. If no axis is passed, one will be created.
    plot_type : str in {"heatmap", "scattermap"} (default="heatmap")
        "heatmap" will draw the matrix using :func:`seaborn.heatmap`, "scattermap" will
        draw each nonzero element of the matrix using :func:`seaborn.scatterplot`.
        "scattermap" is recommended for larger sparse matrices.
    {row,col}_meta : pd.DataFrame or None, (default=None)
        Metadata of the matrix.

        - ``{row,col}_meta`` is pd.DataFrame
            All sorting keywords should only be str or list of str.
            They should contain references to columns in ``meta``.
        - ``{row,col}_meta`` is None
            All sorting keywords should only array-like with the same length as the
            corresponding axis of ``data``.
    {row,col}_group : str, list of str, or array-like, (default=None)
        Attribute(s) by which to group rows/columns of ``data``. If multiple groups are
        specified, rows/columns will be sorted hierarchically (first by the first group),
        then within that group by a possible second group, etc.). Behaves similarly to
        :meth:`pandas.DataFrame.sort_values`.
    {row,col}_group_order : str, list of str, or array-like, (default="size")
        Attribute(s) by which to sort the groups if provided by ``{row,col}_group``.
        Groups are sorted by the mean of this attribute in ascending order. "size" is a
        special keyword which will sort groups by the number of elements per group.
    {row,col}_item_order : str, list of str, or array-like (default=None)
        Attribute(s) by which to sort the individual rows/columns within each group.
    {row,col}_color : str, list of str, or array-like (default=None)
        Attribute(s) to use for drawing colors on the borders of the matrix plot.
    {row,col}_highlight : str, list of str, or array-like (default=None)
        Attribute(s) in meta by which to draw highlighted separators between specific
        groups, can be useful for emphasizing a particular region of the plot. Styling
        of the highlighted separators can be specified via `spinestyle_kws`.
    {row,col}_palette : str, dict, list of str or dict (default="tab10")
        Colormap(s) of the color axes if specified by ``{row,col}_color``.
    {row,col}_ticks : bool, optional (default=True)
        Whether the plot has labels for the groups specified by ``{row,col}_group``.
    {row,col}_tick_pad : int or float (default=None)
        Custom padding to use for the distance between tick axes
    {row,col}_color_pad : int or float (default=None)
        Custom padding to use for the distance between color axes
    border : bool (default=True)
        Whether the plot should have a border.
    center : int (default=0)
        The value at which to center the colormap when plotting divergent data (only
        used when ``plot_type="heatmap"``).
    cmap : str (default="RdBu_r")
        Colormap of the heatmap (only used when ``plot_type="heatmap"``).
    sizes : tuple (default=(5, 10))
        Min and max sizes of dots (only used when ``plot_type="scattermap"``).
    square : bool (default=False)
        Whether the plot should be square.
    gridline_kws : dict (default=None)
        Plotting arguments for the separators.
    spinestyle_kws : dict (default=None)
        Plotting arguments for the spine border.
    highlight_kws : dict (default=None)
        Plotting arguments for the highlighted separators.
    **kwargs
        Additional plotting arguments passed down to the plotting function which will
        draw the matrix ``data``, see ``plot_type`` for more information.

    Returns
    -------
    ax : matplotlib axes object
        Axes in which to draw the plot, by default None
    divider : AxesLocator
        Divider used to add new axes to the plot
    """

    # check for the plot type
    plot_type_opts = ["scattermap", "heatmap"]
    if plot_type not in plot_type_opts:
        raise ValueError(f"`plot_type` must be one of {plot_type_opts}")

    # check for the type and dimension of the data
    _check_data(data, plot_type)

    # check for the types of the sorting arguments
    (
        col_meta,
        col_group,
        col_group_order,
        col_item_order,
        col_color,
        col_highlight,
    ) = _check_sorting_kws(
        data.shape[0],
        col_meta,
        col_group,
        col_group_order,
        col_item_order,
        col_color,
        col_highlight,
    )

    (
        row_meta,
        row_group,
        row_group_order,
        row_item_order,
        row_color,
        row_highlight,
    ) = _check_sorting_kws(
        data.shape[0],
        row_meta,
        row_group,
        row_group_order,
        row_item_order,
        row_color,
        row_highlight,
    )

    # check for the types of the plotting arguments
    col_palette = _check_plotting_kws(col_palette, col_color)
    row_palette = _check_plotting_kws(row_palette, row_color)

    # sort the data and the metadata according to the sorting method
    col_inds, col_meta = sort_meta(
        data.shape[0],
        col_meta,
        col_group,
        group_order=col_group_order,
        item_order=col_item_order,
    )
    row_inds, row_meta = sort_meta(
        data.shape[0],
        row_meta,
        row_group,
        group_order=row_group_order,
        item_order=row_item_order,
    )
    data = data[np.ix_(row_inds, col_inds)]

    # draw the main heatmap/scattermap
    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(10, 10))

    if plot_type == "heatmap":
        sns.heatmap(
            data,
            cmap=cmap,
            ax=ax,
            center=center,
            cbar=False,
            xticklabels=False,
            yticklabels=False,
            **kws,
        )
    elif plot_type == "scattermap":
        scattermap(data, ax=ax, sizes=sizes, **kws)

    # extra features of the graph
    if square:
        ax.axis("square")
    if plot_type == "scattermap":
        ax_pad = 0.5
    else:
        ax_pad = 0
    ax.set_ylim(data.shape[0] - ax_pad, 0 - ax_pad)
    ax.set_xlim(0 - ax_pad, data.shape[1] - ax_pad)

    # make_axes_locatable allows us to create new axes like colormap
    divider = make_axes_locatable(ax)

    # draw separators
    # draw top separators
    draw_separators(
        ax,
        ax_type="x",
        meta=col_meta,
        group=col_group,
        plot_type=plot_type,
        gridline_kws=gridline_kws,
    )
    # draw left separators
    draw_separators(
        ax,
        ax_type="y",
        meta=row_meta,
        group=row_group,
        plot_type=plot_type,
        gridline_kws=gridline_kws,
    )

    # draw highlighted separators
    if highlight_kws is None:
        highlight_kws = dict(color="black", linestyle="-", linewidth=1)
    # draw top highlights
    if col_highlight is not None:
        draw_separators(
            ax,
            ax_type="x",
            meta=col_meta,
            group=col_highlight,
            plot_type=plot_type,
            gridline_kws=highlight_kws,
        )
    # draw left highlights
    if row_highlight is not None:
        draw_separators(
            ax,
            ax_type="y",
            meta=row_meta,
            group=row_highlight,
            plot_type=plot_type,
            gridline_kws=highlight_kws,
        )

    # these two variables are used to identify the first axes
    col_first_ax = True
    row_first_ax = True
    # draw colors
    # draw top colors
    if len(col_color) > 0:
        if col_color_pad is None:
            col_color_pad = len(col_color) * [0.5]
        if col_first_ax:
            col_color_pad[0] = 0
            col_first_ax = False

        rev_color = list(col_color[::-1])
        if len(col_palette) > 1:
            rev_color_palette = list(col_palette[::-1])
        else:
            rev_color_palette = list(col_palette * len(col_color))

        for i, sc in enumerate(rev_color):
            color_ax = divider.append_axes(
                "top", size="3%", pad=col_color_pad[i], sharex=ax
            )
            _remove_shared_ax(color_ax)
            draw_colors(
                color_ax,
                meta=col_meta,
                divider=divider,
                ax_type="x",
                color=rev_color[i],
                palette=rev_color_palette[i],
            )
    # draw left colors
    if len(row_color) > 0:
        if row_color_pad is None:
            row_color_pad = len(row_color) * [0.5]
        if row_first_ax:
            row_color_pad[0] = 0
            row_first_ax = False

        rev_color = list(row_color[::-1])
        if len(row_palette) > 1:
            rev_color_palette = list(row_palette[::-1])
        else:
            rev_color_palette = list(row_palette * len(row_color))

        for i, sc in enumerate(rev_color):
            color_ax = divider.append_axes(
                "left", size="3%", pad=row_color_pad[i], sharey=ax
            )
            _remove_shared_ax(color_ax)
            draw_colors(
                color_ax,
                meta=row_meta,
                divider=divider,
                ax_type="y",
                color=rev_color[i],
                palette=rev_color_palette[i],
            )

    # draw ticks
    # draw top ticks
    if len(col_group) > 0 and col_ticks:
        if col_tick_pad is None:
            col_tick_pad = len(col_group) * [0.8]
        if col_first_ax:
            col_tick_pad[0] = 0
            col_first_ax = False
        else:
            col_tick_pad[0] = 0.5

        # Reverse the order of the group class, so the ticks are drawn
        # in the opposite order
        rev_group = list(col_group[::-1])
        for i, sc in enumerate(rev_group):

            # Add a new axis when needed
            tick_ax = divider.append_axes(
                "top", size="1%", pad=col_tick_pad[i], sharex=ax
            )
            _remove_shared_ax(tick_ax)
            tick_ax = _set_spines(
                tick_ax, left=False, right=False, top=True, bottom=False
            )

            # Draw the ticks for the x axis
            draw_ticks(
                tick_ax,
                ax_type="x",
                meta=col_meta,
                group=rev_group[i:],
                plot_type=plot_type,
            )
            ax.xaxis.set_label_position("top")
    # draw left ticks
    if len(row_group) > 0 and row_ticks:
        if row_tick_pad is None:
            row_tick_pad = len(row_group) * [0.8]
        if row_first_ax:
            row_tick_pad[0] = 0
            row_first_ax = False
        else:
            row_tick_pad[0] = 0.5

        # Reverse the order of the group class, so the ticks are drawn
        # in the opposite order
        rev_group = list(row_group[::-1])
        for i, sc in enumerate(rev_group):
            # Add a new axis when needed
            tick_ax = divider.append_axes(
                "left", size="1%", pad=row_tick_pad[i], sharey=ax
            )
            _remove_shared_ax(tick_ax)
            tick_ax = _set_spines(
                tick_ax, left=True, right=False, top=False, bottom=False
            )

            # Draw the ticks for the y axis
            draw_ticks(
                tick_ax,
                ax_type="y",
                meta=row_meta,
                group=rev_group[i:],
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


def adjplot(  # type: ignore
    data,
    ax=None,
    meta=None,
    plot_type="heatmap",
    group=None,
    group_order="size",
    item_order=None,
    color=None,
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
    **kws,
):
    r"""
    Sorts and plots a square matrix in various ways, and with optional information added
    to the margin of the matrix plot.

    This function is a wrapper around ``matrixplot`` which assumes that the plotted
    matrix is square, and that the rows and columns represent the same items and have
    the same metadata (e.g. the adjacency matrix of a graph).

    Read more in the `Tutorial https://microsoft.github.io/graspologic/tutorials/plotting/matrixplot.html`

    Parameters
    ----------
    data : np.ndarray or scipy.sparse.csr_matrix with ndim=2
        Matrix to plot, must be square.
        Sparse matrix input is only accepted if ``plot_type == 'scattermap'``.
    ax : matplotlib axes object (default=None)
        Axes in which to draw the plot. If no axis is passed, one will be created.
    plot_type : str in {"heatmap", "scattermap"} (default="heatmap")
        "heatmap" will draw the matrix using :func:`seaborn.heatmap`, "scattermap" will
        draw each nonzero element of the matrix using :func:`seaborn.scatterplot`.
        "scattermap" is recommended for larger sparse matrices.
    meta : pd.DataFrame or None, (default=None)
        Metadata of the matrix.

        - ``meta`` is pd.DataFrame
            All sorting keywords should only be str or list of str.
            They should contain references to columns in ``meta``.
        - ``meta`` is None
            All sorting keywords should only array-like with the same length as the
            corresponding axis of ``data``.
    group : str, list of str, or array-like, (default=None)
        Attribute(s) by which to group rows/columns of ``data``. If multiple groups are
        specified, rows/columns will be sorted hierarchically (first by the first group),
        then within that group by a possible second group, etc.). Behaves similarly to
        :meth:`pandas.DataFrame.sort_values`.
    group_order : str, list of str, or array-like, (default="size")
        Attribute(s) by which to sort the groups if provided by ``group``.
        Groups are sorted by the mean of this attribute in ascending order. "size" is a
        special keyword which will sort groups by the number of elements per group.
    item_order : str, list of str, or array-like (default=None)
        Attribute(s) by which to sort the individual rows/columns within each group.
    color : str, list of str, or array-like (default=None)
        Attribute(s) to use for drawing colors on the borders of the matrix plot.
    highlight : str, list of str, or array-like (default=None)
        Attribute(s) in meta by which to draw highlighted separators between specific
        groups, can be useful for emphasizing a particular region of the plot. Styling
        of the highlighted separators can be specified via `spinestyle_kws`.
    palette : str, dict, list of str or dict (default="tab10")
        Colormap(s) of the color axes if specified by ``color``.
    ticks : bool, optional (default=True)
        Whether the plot has labels for the groups specified by ``group``.
    tick_pad : int or float (default=None)
        Custom padding to use for the distance between tick axes
    color_pad : int or float (default=None)
        Custom padding to use for the distance between color axes
    border : bool (default=True)
        Whether the plot should have a border.
    center : int (default=0)
        The value at which to center the colormap when plotting divergent data (only
        used when ``plot_type="heatmap"``).
    cmap : str (default="RdBu_r")
        Colormap of the heatmap (only used when ``plot_type="heatmap"``).
    sizes : tuple (default=(5, 10))
        Min and max sizes of dots (only used when ``plot_type="scattermap"``).
    square : bool (default=False)
        Whether the plot should be square.
    gridline_kws : dict (default=None)
        Plotting arguments for the separators.
    spinestyle_kws : dict (default=None)
        Plotting arguments for the spine border.
    highlight_kws : dict (default=None)
        Plotting arguments for the highlighted separators.
    **kwargs
        Additional plotting arguments passed down to the plotting function which will
        draw the matrix ``data``, see ``plot_type`` for more information.

    Returns
    -------
    ax : matplotlib axes object
        Axes in which to draw the plot, by default None
    divider : AxesLocator
        Divider used to add new axes to the plot
    """
    outs = matrixplot(
        data,
        ax=ax,
        col_meta=meta,
        row_meta=meta,
        plot_type=plot_type,
        col_group=group,
        row_group=group,
        col_group_order=group_order,
        row_group_order=group_order,
        col_item_order=item_order,
        row_item_order=item_order,
        col_color=color,
        row_color=color,
        col_highlight=highlight,
        row_highlight=highlight,
        col_palette=palette,
        row_palette=palette,
        col_ticks=ticks,
        row_ticks=ticks,
        col_tick_pad=tick_pad,
        row_tick_pad=tick_pad,
        col_color_pad=color_pad,
        row_color_pad=color_pad,
        border=border,
        center=center,
        cmap=cmap,
        sizes=sizes,
        square=square,
        gridline_kws=gridline_kws,
        spinestyle_kws=spinestyle_kws,
        highlight_kws=highlight_kws,
        **kws,
    )

    return outs
