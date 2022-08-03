from scipy.sparse import csr_matrix

from .sbm import group_connection_test, group_connection_test_paired
from .er import erdos_renyi_test, erdos_renyi_test_paired

import numpy as np

A1 = np.array([[0,1,0,0,1],[0,0,0,1,1],[1,1,0,1,1],[1,0,1,0,1],[0,0,0,0,1]],np.int32)
A2 = np.array([[0,1,1,1,1],[1,0,1,0,1],[1,1,1,0,1],[0,1,0,1,0],[1,0,1,0,1]], np.int32)
labels1 = np.array([1,2,1,1,2], np.int32)
labels2 = np.array([1,2,1,1,2], np.int32)
stat,pvalue,misc = group_connection_test(A1,A2,labels1,labels2)

print(pvalue)