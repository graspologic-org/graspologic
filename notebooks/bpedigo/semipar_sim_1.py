import numpy as np
from graspy.inference import SemiparametricTest
from graspy.embed import AdjacencySpectralEmbed, select_dimension
import warnings
from graspy.simulations import sbm
from graspy.utils import symmetrize
import time
import sys
import getopt
import pickle  
from multiprocessing import Pool, cpu_count
from functools import partial

B_BASE = np.array([[0.5, 0.2],
                  [0.2, 0.5]])

def get_block_probs(eps):
    B = B_BASE.copy()
    B[0,0] += eps
    B[1,1] += eps
    return B

def run_sim(Bx, By, n_components, n_bootstraps, sizes, seed):
    np.random.seed(seed)
    A0 = sbm(sizes,Bx, loops=False)
    A1 = sbm(sizes,Bx, loops=False)
    A2 = sbm(sizes,By, loops=False)
    spt_null = SemiparametricTest(n_components=n_components, 
                                n_bootstraps=n_bootstraps)
    spt_alt = SemiparametricTest(n_components=n_components, 
                                n_bootstraps=n_bootstraps)
    spt_null.fit(A0, A1)
    spt_alt.fit(A0, A2)
    return (spt_null, spt_alt)

def main(argv):
    t = time.clock()
    np.random.seed(83450277)
    opts, args = getopt.getopt(argv, 'b:s:n:c:')
    n_components = None
    for opt, arg in opts:
        if opt == '-b':
            n_bootstraps = int(arg)
            print('Bootstraps for each trial: {}'.format(n_bootstraps))
        elif opt == '-s':
            size = int(arg)
            print('Size of each community: {}'.format(size))
        elif opt == '-n':
            n_sims = int(arg)
            print('Number of simulation loops: {}'.format(n_sims))
        elif opt == '-c':
            n_components = int(arg)
            print('Number of embedded dimensions: {}'.format(n_components))
    print('Seeing {} CPUs'.format(cpu_count()))
    
    sizes = 2*[size]
    Bx = get_block_probs(0)
    epsilons = [0, 0.05, 0.1, 0.2]
    # epsilons = [0, 0.2]
    epsilon_outputs = []
    for eps in epsilons:
        print('Epsilon = {}'.format(eps))
        By = get_block_probs(eps)
        seeds = [np.random.randint(1, 100000000) for _ in range(n_sims)]
        run_sim_partial = partial(run_sim, Bx, By, n_components, n_bootstraps, sizes,)
        with Pool(cpu_count() - 1) as p:
            outputs = p.map(run_sim_partial, seeds)
        epsilon_outputs.append(outputs)
            
    params = {
              'comm_size':size,
              'n_sims':n_sims,
              'n_components':n_components,
              'n_bootstraps':n_bootstraps,
              'epsilons':epsilons,
              'B_base':B_BASE
              }
    took = time.clock() - t
    output = {
              'test_by_epsilon':epsilon_outputs,
              'params':params,
              'time':took
              }

    print('Took {}'.format(took))
    job_id = str(hash(time.time()))
    print('File name: {}.pickle'.format(job_id))
    f = open(job_id + '.pickle', 'wb')
    pickle.dump(output, f)
    f.close()

if __name__ == '__main__':
    main(sys.argv[1:])