import numpy as np
from graspy.inference import SemiparametricTest
from graspy.embed import AdjacencySpectralEmbed, select_dimension
import warnings
from graspy.simulations import binary_sbm, rdpg_from_latent
from graspy.utils import symmetrize
import time
import sys
import getopt
import pickle  
from multiprocessing import Pool, cpu_count

# f = open('2110072216651244771.pickle', 'rb')
# f = open('931210899241672712.pickle', 'rb')
# f = open('476622349264429854.pickle', 'rb')
f = open('611522530389271978.pickle', 'rb')
d = pickle.load(f)
f.close()
print(d)
