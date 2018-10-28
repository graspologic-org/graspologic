import sys
import matplotlib as mpl
if sys.platform == 'darwin':
    mpl.use('tkAgg')
elif sys.platform == 'linux':
    mpl.use('Agg')

from .plot import heatmap, grid_plot

__all__ = [heatmap, grid_plot]