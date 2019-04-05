import numpy as np
np.random.seed(88889999)
import graspy
from graspy.inference import SemiparametricTest
from graspy.embed import AdjacencySpectralEmbed
from graspy.simulations import sbm, rdpg
from graspy.utils import symmetrize
from graspy.plot import heatmap, pairplot

import numpy as np
from mgcpy.independence_tests.dcorr import DCorr
from sklearn import preprocessing
import mgcpy

from tqdm import tqdm

def triu_no_diag(A):
    '''
    Get the entries in the upper triangular part of the adjacency matrix (not
    including the diagonal)
    Returns
    --------
    2d array:
        The vectorized upper triangular part of graph A
    '''
    n = A.shape[0]
    iu1 = np.triu_indices(n, 1)
    triu_vec = A[iu1]
    return triu_vec[:, np.newaxis]

def permute_matrix(A):
    permuted_indices = np.random.permutation(np.arange(A.shape[0]))
    A = A[np.ix_(permuted_indices, permuted_indices)]
    return A

def k_sample_transform(x, y, is_y_categorical=False):
    if not is_y_categorical:
        u = np.concatenate([x, y], axis=0)
        v = np.concatenate([np.repeat(1, x.shape[0]), np.repeat(2, y.shape[0])], axis=0)
    else:
        u = x
        v = preprocessing.LabelEncoder().fit_transform(y.flatten()) + 1

    if len(u.shape) == 1:
        u = u[..., np.newaxis]
    if len(v.shape) == 1:
        v = v[..., np.newaxis]

    return u, v

def paired_two_sample_transform(x, y):
    joint_distribution = np.concatenate([x, y], axis=0)  # (2n, p) shape

    pairwise_sampled_xy = np.array([joint_distribution[np.random.randint(joint_distribution.shape[0], size=2), :]
                                    for _ in range(x.shape[0])])  # (n, 2, p) shape
    pairwise_sampled_x = pairwise_sampled_xy[:, 0]  # (n, p) shape
    pairwise_sampled_y = pairwise_sampled_xy[:, 1]  # (n, p) shape

    # compute the eucledian distances
    randomly_sampled_pairs_distance = np.linalg.norm(pairwise_sampled_x - pairwise_sampled_y, axis=1)
    actual_pairs_distance = np.linalg.norm(x - y, axis=1)

    u, v = k_sample_transform(randomly_sampled_pairs_distance, actual_pairs_distance)

    return u, v


def power(indept_test, sample_func, transform_func, mc=100, alpha=0.05,
          is_null=False, **kwargs):
    test_stat_null_array = np.zeros(mc)
    test_stat_alt_array = np.zeros(mc)
    for i in range(mc):
        A, B = sample_func(**kwargs)
        if is_null:
            A = permute_matrix(A)
        test_stat_alt, _ = indept_test.test_statistic(
            matrix_X=transform_func(A), matrix_Y=transform_func(B), is_fast=True)
        test_stat_alt_array[i] = test_stat_alt

        # generate the null by permutation
        A_null = permute_matrix(A)
        test_stat_null, _ = indept_test.test_statistic(
            matrix_X=transform_func(A_null), matrix_Y=transform_func(B), is_fast=True)
        test_stat_null_array[i] = test_stat_null
    # if pearson, use the absolute value of test statistic then use one-sided
    # rejection region
    if indept_test.get_name() == 'pearson':
        test_stat_null_array = np.absolute(test_stat_null_array)
        test_stat_alt_array = np.absolute(test_stat_alt_array)
    critical_value = np.sort(test_stat_null_array)[math.ceil((1-alpha)*mc)]
    power = np.where(test_stat_alt_array > critical_value)[0].shape[0] / mc
    return power

def get(n=50):
    ns = [n, n]
    p1 = np.array([[.9,.1],[.1,.9]])
    p2 = np.array([[.9,.1],[.1,.9]])
    A1 = sbm(ns,p1)
    A2 = sbm(ns,p2)
    X1 = AdjacencySpectralEmbed().fit_transform(A1)
    X2 = AdjacencySpectralEmbed().fit_transform(A2)
    return X1, X2

pows = []
x = range(10,101,10)
for n in x:
    mgc = mgcpy.independence_tests.mgc.mgc()
    power(mgc, get(n=n), paired_two_sample_transform())
    p.append(power)

import matplotlib.pyplot as plt
plt.plot(x,pows)
plt.plot(x,[1]*len(x),'r-.',alpha=0.8)
plt.xlabel('n')
plt.ylabel('power')
plt.savefig('power_curve_mgc.png')
plt.show()
