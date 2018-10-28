import numpy as np
import pytest

from graspy.plot.plot import heatmap, gridplot
from graspy.simulations.simulations import er_np


def test_heatmap_inputs():
    """
    test parameter checks
    """
    X = np.random.rand(10, 10)

    with pytest.raises(TypeError):
        heatmap(X='input')

    # transform
    with pytest.raises(ValueError):
        transform = 'bad transform'
        heatmap(X, transform=transform)

    # figsize
    with pytest.raises(TypeError):
        figsize = 'bad figsize'
        heatmap(X, figsize=figsize)

    # title
    with pytest.raises(TypeError):
        title = 1
        heatmap(X, title=title)

    # context
    with pytest.raises(TypeError):
        context = 1
        heatmap(X, context=context)

    with pytest.raises(ValueError):
        context = 'journal'
        heatmap(X, context=context)

    # font_scale
    with pytest.raises(TypeError):
        font_scale = '1'
        heatmap(X, font_scale=font_scale)

    # ticklabels
    with pytest.raises(TypeError):
        xticklabels = 'labels'
        yticklabels = 'labels'
        heatmap(X, xticklabels=xticklabels, yticklabels=yticklabels)

    with pytest.raises(ValueError):
        xticklabels = ['{}'.format(i) for i in range(5)]
        yticklabels = ['{}'.format(i) for i in range(5)]
        heatmap(X, xticklabels=xticklabels, yticklabels=yticklabels)

    # cmap
    with pytest.raises(TypeError):
        cmap = 123
        heatmap(X, cmap=cmap)

    # center
    with pytest.raises(TypeError):
        center = 'center'
        heatmap(X, center=center)

    # cbar
    with pytest.raises(TypeError):
        cbar = 1
        heatmap(X, cbar=cbar)


def test_heatmap_output():
    """
    simple function to see if plot is made without errors
    """
    X = er_np(10, .5)
    xticklabels = ['Dimension {}'.format(i) for i in range(10)]
    yticklabels = ['Dimension {}'.format(i) for i in range(10)]

    fig = heatmap(
        X, transform='log', xticklabels=xticklabels, yticklabels=yticklabels)
    fig = heatmap(X, transform='zero-boost')
    fig = heatmap(X, transform='simple-all')
    fig = heatmap(X, transform='simple-nonzero')
    fig = heatmap(X, cmap='gist_rainbow')


def test_grid_plot_inputs():
    X = er_np(10, .5)
    labels = ['ER(10, 0.5)']

    with pytest.raises(TypeError):
        grid_plot(X='input', labels=labels)

    with pytest.raises(ValueError):
        grid_plot(X, labels=['a', 'b'])

    # transform
    with pytest.raises(ValueError):
        transform = 'bad transform'
        grid_plot(X, labels=labels, transform=transform)

    # height
    with pytest.raises(TypeError):
        height = 'bad figsize'
        grid_plot(X, labels=labels, height=height)

    # title
    with pytest.raises(TypeError):
        title = 1
        grid_plot(X, labels=labels, title=title)

    # context
    with pytest.raises(TypeError):
        context = 1
        heatmap(X, context=context)

    with pytest.raises(ValueError):
        context = 'journal'
        heatmap(X, context=context)

    # font_scale
    with pytest.raises(TypeError):
        font_scale = '1'
        heatmap(X, font_scale=font_scale)