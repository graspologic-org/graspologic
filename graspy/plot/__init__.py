import sys

# Handle matplotlib backend for different operating systems
import matplotlib as mpl

if sys.platform == "darwin":
    mpl.use("tkAgg")
elif sys.platform == "linux":
    mpl.use("Agg")

from .plot import heatmap, gridplot, pairplot, degreeplot, edgeplot, screeplot

__all__ = ["heatmap", "gridplot", "pairplot", "degreeplot", "edgeplot", "screeplot"]
