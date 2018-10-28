import sys
if sys.platform == 'darwin':
    import matplotlib as mpl
    mpl.use('tkAgg')

from .plot import heatmap, grid_plot

__all__ = [heatmap, grid_plot]