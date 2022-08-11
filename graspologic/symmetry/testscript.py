from scipy.sparse import csr_matrix

from graspologic.bilateral_connectome.sbm import group_connection_test, group_connection_test_paired
from graspologic.bilateral_connectome.er import erdos_renyi_test, erdos_renyi_test_paired

import numpy as np

A1 = np.array([[0,1,0,0,1],[0,0,0,1,1],[1,1,0,1,1],[1,0,1,0,1],[0,0,0,0,1]],np.int32)
sA1 = csr_matrix(A1)
A2 = np.array([[0,1,1,1,1],[1,0,1,0,1],[1,1,1,0,1],[0,1,0,1,0],[1,0,1,0,1]], np.int32)
sA2 = csr_matrix(A2)
labels1 = np.array([1,2,1,1,2], np.int32)
labels2 = np.array([1,2,1,1,2], np.int32)
stat,pvalue,misc = group_connection_test(sA1,sA2,labels1,labels2)

print(pvalue)

stat,pvalue,misc=group_connection_test_paired(sA1,sA2,labels1)
print(pvalue)