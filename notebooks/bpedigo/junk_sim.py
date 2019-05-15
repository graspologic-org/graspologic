import numpy as np
from graspy.inference import SemiparametricTest
from graspy.embed import AdjacencySpectralEmbed, select_dimension
import warnings

# from graspy.simulations import sbm
from graspy.utils import symmetrize
import time
import sys
import getopt
import pickle
from multiprocessing import Pool, cpu_count


def junk(seed,):
    np.random.seed(seed)
    msg = np.random.normal(0, 1)
    return msg
    # outputs = [p.get() for p in tests]
    # epsilon_outputs.append(outputs)


if __name__ == "__main__":
    p = Pool()
    n_sims = 100
    seeds = [np.random.randint(1, 100000000) for _ in range(n_sims)]
    print(seeds)
    jobs = [p.apply_async(junk, args=(seeds[i],)) for i in range(n_sims)]
    outputs = [p.get() for p in jobs]
    print(outputs)
