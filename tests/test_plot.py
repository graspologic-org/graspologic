import numpy as np
import pytest

from graspy.plot.plot import heatmap, gridplot, pairplot
from graspy.simulations.simulations import er_np


def test_common_inputs():
    X = er_np(100, 0.5)
    grid_labels = ['Test1']

    # test figsize
    with pytest.raises(TypeError):
        figsize = 'bad figsize'
        heatmap(X, figsize=figsize)

    # test height
    height = '1'
    with pytest.raises(TypeError):
        gridplot([X], grid_labels, height=height)
    with pytest.raises(TypeError):
        pairplot(X, height=height)

    # test title
    title = 1
    with pytest.raises(TypeError):
        heatmap(X, title=title)
    with pytest.raises(TypeError):
        gridplot([X], grid_labels, title=title)
    with pytest.raises(TypeError):
        pairplot(X, title=title)

    # test context
    context = 123
    with pytest.raises(TypeError):
        heatmap(X, context=context)
    with pytest.raises(TypeError):
        gridplot([X], grid_labels, context=context)
    with pytest.raises(TypeError):
        pairplot(X, context=context)

    context = 'journal'
    with pytest.raises(ValueError):
        heatmap(X, context=context)
    with pytest.raises(ValueError):
        gridplot([X], grid_labels, context=context)
    with pytest.raises(ValueError):
        pairplot(X, context=context)

    # test font scales
    font_scales = ['1', []]
    for font_scale in font_scales:
        with pytest.raises(TypeError):
            heatmap(X, font_scale=font_scale)
        with pytest.raises(TypeError):
            gridplot([X], grid_labels, font_scale=font_scale)
        with pytest.raises(TypeError):
            pairplot(X, cont_scale=font_scale)

    # ticklabels
    with pytest.raises(TypeError):
        xticklabels = 'labels'
        yticklabels = 'labels'
        heatmap(X, xticklabels=xticklabels, yticklabels=yticklabels)

    with pytest.raises(ValueError):
        xticklabels = ['{}'.format(i) for i in range(5)]
        yticklabels = ['{}'.format(i) for i in range(5)]
        heatmap(X, xticklabels=xticklabels, yticklabels=yticklabels)


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


def test_gridplot_inputs():
    X = [er_np(10, .5)]
    labels = ['ER(10, 0.5)']

    with pytest.raises(TypeError):
        gridplot(X='input', labels=labels)

    with pytest.raises(ValueError):
        gridplot(X, labels=['a', 'b'])

    # transform
    with pytest.raises(ValueError):
        transform = 'bad transform'
        gridplot(X, labels=labels, transform=transform)


def test_gridplot_outputs():
    """
    simple function to see if plot is made without errors
    """
    X = [er_np(10, .5) for _ in range(2)]
    labels = ['Random A', 'Random B']
    fig = gridplot(X, labels)
    fig = gridplot(X, labels, transform='zero-boost')
    fig = gridplot(X, labels, 'simple-all', title='Test', font_scale=.9)


def test_pairplot_inputs():
    X = np.random.rand(15, 3)
    Y = ['A'] * 5 + ['B'] * 5 + ['C'] * 5

    # test data
    with pytest.raises(TypeError):
        pairplot(X='test')

    with pytest.raises(ValueError):
        pairplot(X=X, Y=['A'])

    with pytest.raises(TypeError):
        pairplot(X, col_names='A')

    with pytest.raises(ValueError):
        pairplot(X, col_names=['1', '2'])

    with pytest.raises(ValueError):
        pairplot(X, col_names=['1', '2', '3'], variables=[1, 2, 3, 4])

    with pytest.raises(KeyError):
        pairplot(X, col_names=['1', '2', '3'], variables=['A', 'B'])


def test_pairplot_outputs():
    X = np.random.rand(15, 3)
    Y = ['A'] * 5 + ['B'] * 5 + ['C'] * 5
    col_names = ['Feature1', 'Feature2', 'Feature3']

    fig = pairplot(X)
    fig = pairplot(X, Y)
    fig = pairplot(X, Y, col_names)
    fig = pairplot(
        X,
        Y,
        col_names,
        title='Test',
        height=1.5,
        variables=['Feature1', 'Feature2'])
