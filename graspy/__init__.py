import warnings

import graspy.cluster
import graspy.datasets
import graspy.embed
import graspy.inference
import graspy.models
import graspy.pipeline
import graspy.plot
import graspy.simulations
import graspy.utils

warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
warnings.simplefilter("always", category=UserWarning)


__version__ = "0.1"
