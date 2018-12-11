#%%
import numpy as np
from graspy.inference import SemiparametricTest
from graspy.embed import AdjacencySpectralEmbed, select_dimension
import warnings
from graspy.utils import symmetrize
import time
import sys
import getopt
import pickle  
from multiprocessing import Pool, cpu_count
import seaborn as sns
from pathlib import Path
# f = open('2110072216651244771.pickle', 'rb')
# f = open('931210899241672712.pickle', 'rb')
# f = open('476622349264429854.pickle', 'rb')
#%%
picklePath = Path('./pickles')
f = open(picklePath / '127933127237990926.pickle', 'rb')
# f = open(picklePath / '10624032647480697.pickle', 'rb')
d = pickle.load(f)
f.close()
print(d.keys())
print(d['test_by_epsilon'][0][0])
#%%
tests_by_epsilon = d['test_by_epsilon']
test_null_p = np.zeros((len(tests_by_epsilon[0]),2))
test_alt_p = np.zeros((len(tests_by_epsilon[0]),2))
sig = 0.05
null_rej = 0
alt_rej = 0
for e, test_epsilon in enumerate(tests_by_epsilon):
    for t, test_group in enumerate(test_epsilon):
        test_null_p[t,e] = test_group[0].p
        test_alt_p[t,e] = test_group[1].p
        null_rej += test_group[0].p < sig
        alt_rej += test_group[1].p < sig
e = 1
print(np.sum(test_null_p[:,e] < sig) / 200)
print(np.sum(test_alt_p[:,e] < sig) / 200)

#%%
sns.distplot(test_alt_p[:,1])

#%%
print('Null mean')
print(np.mean(test_null_p, axis=0))
print('Alt mean')
print(np.mean(test_alt_p, axis=0))
print(np.var(test_null_p, axis=0))
print(np.var(test_alt_p, axis=0))
print(test_null_p)
print(test_alt_p)
#%%

sns.distplot(tests_by_epsilon[0][0][0].T1_bootstrap)
sns.distplot(tests_by_epsilon[0][0][0].T2_bootstrap)
plt.show()
sns.distplot(tests_by_epsilon[0][0][1].T1_bootstrap)
sns.distplot(tests_by_epsilon[0][0][1].T2_bootstrap)
tests_by_epsilon[0][0][1].T_sample
tests_by_epsilon[0][0][1].p
#%%
tests_by_epsilon[1][0]
sns.distplot(tests_by_epsilon[1][0][0].T1_bootstrap)
sns.distplot(tests_by_epsilon[1][0][0].T2_bootstrap)
plt.show()
sns.distplot(tests_by_epsilon[1][0][1].T1_bootstrap)
sns.distplot(tests_by_epsilon[1][0][1].T2_bootstrap)
tests_by_epsilon[1][0][1].T_sample
#%%
for i, tests in enumerate(tests_by_epsilon):
    print(i)
    print(tests)