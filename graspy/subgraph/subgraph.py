# Copyright 2019 NeuroData (http://neurodata.io)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np

from mgcpy.independence_tests.mgc import MGC
from mgc.independence import Dcorr, RV, CCA


def non_iterative_screen(a_tensor, y_labels, c, opt):
    """
    Performs non-iterative screening on graphs to estimate signal subgraph.
    
    Parameters
    ----------
    a_tensor: np.ndarray, shape (n_graphs, n_vertices, n_vertices)
        Tensor of adjacency matrices
    y_labels: np.ndarray, shape (n_graphs, 1)
        Vector of ground truth labels
    c: float
        Correlation threshold value chosen by user
    opt: string
        Indication of which statistic to use
    
    Returns
    -------
    corrs: np.ndarray, shape (n_vertices, 1)
        Vector of correlation values for each node
    S_hat: np.ndarray, shape (ss_size, 1)
        Estimated signal subgraph, approximated with non-iterative
        vertex screening.
        
    References
    ----------
    .. [1] S. Wang, C. Chen, A. Badea, Priebe, C.E., Vogelstein, J.T.  "Signal 
    Subgraph Estimation Via Vertex Screening," arXiv: 1801.07683 [stat.ME], 2018
    """
    if type(a_tensor) is not np.ndarray:
        raise TypeError("a_tensor must be numpy.ndarray")
    if a_tensor.shape[1] != a_tensor.shape[2]:
        raise ValueError("Entries in a_tensor must be square matricies")
    if len(a_tensor.shape) != 3:
        raise ValueError("a_tensor must be a tensor")
    # Import needed statistical modules from mgcpy package
    mgc = MGC()
    dcorr = Dcorr()
    rv = RV()
    cca = CCA()

    # Finding dimension of each matrix
    N = len(a_tensor[0])

    # Create vector of zeros that will become vector of correlations
    corrs = np.zeros((N, 1))

    for i in range(N):

        # Stacks the ith row of each matrix in tensor,
        # creates matrix with dimension len(a_tensor) by N
        mat = a_tensor[:, i]

        # Statistical measurement chosen by the user
        if opt == "mgc":
            c_u_0, independence_test_metadata_0 = mgc.test_statistic(mat, y_labels)
            corrs[i][0] = c_u_0
        elif opt == "dcorr":
            c_u_1 = dcorr._statistic(mat, y_labels)
            corrs[i][0] = c_u_1
        elif opt == "rv":
            c_u_2 = rv._statistic(mat, y_labels)
            corrs[i][0] = c_u_2
        else:
            c_u_3 = cca._statistic(mat, y_labels)
            corrs[i][0] = c_u_3

    # Finds indicies of correlation values greater than c and makes that into column vector
    S_hat = np.arange(N).reshape(N,1)
    ind = corrs > c
    S_hat = S_hat[ind].reshape(len(S_hat[ind]),1)

    return S_hat, corrs


def iterative_screen(a_tensor, y_labels, ss_size, delta, opt):
    """
    Performs iterative screening on graphs.
     
    Parameters
    ----------
    a_tensor: np.ndarray, shape (n_graphs, n_vertices, n_vertices)
        Tensor of adjacency matrices
    y_labels: np.ndarray, shape (n_graphs, 1)
        Vector of ground truth labels
    ss_size: int
        Signal subgraph size specified by the user
    delta:
        Quantile threshold specified by the user
    opt: string
        Indication of which statistic to use
        
    Returns
    -------
    cors: np.ndarray, shape (n_vertices, 1)
        vector of correlation values repeatedly summed for all the nodes
       
    References
    ----------
    .. [1] S. Wang, C. Chen, A. Badea, Priebe, C.E., Vogelstein, J.T.  "Signal 
    Subgraph Estimation Via Vertex Screening," arXiv: 1801.07683 [stat.ME], 2018
    """

    # Get dimensions of tensor
    m = len(a_tensor)
    n = len(a_tensor[0])

    # Create empty array to store correlation values
    cors = np.zeros((n, 1))
    iter = 1

    # Indexing vector
    Sindex = np.arange(n, dtype="int64")

    while ((1 - delta) ** iter) * n > ss_size:

        # Create new Atmp tensor each time as the matrix sizes change every iteration
        dim = len(Sindex)
        Atmp = np.zeros((m, dim, dim))
        for i in range(m):
            Atmp[i] = a_tensor[i][Sindex][:, Sindex]

        # Find correlation values
        vals, tmpcors = non_iterative_screen(Atmp, y_labels, 0, opt)

        # Take specified quantile of correlation values
        tmpq = np.quantile(tmpcors, delta)

        # Add weight to the correlation values that have not been taken out
        cors[Sindex] = tmpcors + iter

        # New Sindex is where the correlation values are greater than the quantile value
        ind = tmpcors > tmpq
        ind = ind.reshape(1, len(ind))
        Sindex = Sindex[ind[0]]
        iter += 1

    return cors
