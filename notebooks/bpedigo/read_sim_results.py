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
f = open('1997605371270819926.pickle', 'rb')
d = pickle.load(f)
f.close()
print(d.keys())
print(d['test_by_epsilon'][0][0])
tests_by_epsilon = d['test_by_epsilon']
test_null_p = np.zeros((1000,4))
test_alt_p = np.zeros((1000,4))
for e, test_epsilon in enumerate(tests_by_epsilon):
    for t, test_group in enumerate(test_epsilon):
        test_null_p[t,e] = test_group[0].p
        test_alt_p[t,e] = test_group[1].p
print(np.mean(test_null_p, axis=0))
print(np.mean(test_alt_p, axis=0))
    
