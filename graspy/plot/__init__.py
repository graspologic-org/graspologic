import sys

# Handle matplotlib backend for different operating systems
import matplotlib as mpl

<<<<<<< HEAD
if sys.platform == "darwin":
    mpl.use("tkAgg")
elif sys.platform == "linux":
    mpl.use("Agg")

from .plot import heatmap, gridplot, pairplot, degreeplot, edgeplot, screeplot

__all__ = ["heatmap", "gridplot", "pairplot", "degreeplot", "edgeplot", "screeplot"]
=======
from .plot import heatmap, gridplot, pairplot, degreeplot, edgeplot, screeplot

__all__ = [
    'heatmap', 'gridplot', 'pairplot', 'degreeplot', 'edgeplot', 'screeplot'
]
>>>>>>> ab53e172669d1c21edbc360b624adb3d8ce33927
