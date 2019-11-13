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
import math
from scipy.optimize import linear_sum_assignment
from scipy.optimize import minimize_scalar
from .skp import SinkhornKnopp



class FastApproximateQAP:
    """
    Fast Approximate QAP Algorithm (FAQ)
    The FAQ algorithm solves the Quadratic Assignment Problem (QAP) finding an alignment of the
    vertices of two graphs which minimizes the number of induced edge disagreements [1].
    
    
    Parameters
    ----------
    
    n_init : int
        Number of random initializations of the starting permutation matrix that the FAQ algorithm will undergo.
        n_init automatically set to 1 if init_method = 'barycenter'
        
    init_method : string
        The initial position chosen
        "barycenter" : the non-informative “flat doubly stochastic matrix,” J=1*1^T/n, i.e the barycenter of
        the feasible region
        "rand" : some random point near J, (J+K)/2, where K is some random doubly stochastic matrix
        
    Attributes
    ----------
    
    perm_inds_ : array, size (n,1) where n is the number of vertices in the graphs fitted
        The indices of the optimal permutation found via FAQ
        
    ofv_ : float
        The objective function value of for the optimal permutation found
        
        
    References
    ----------
    .. [1] J. T. Vogelstein, J. M. Conroy, V. Lyzinski, L. J. Podrazik, S. G. Kratzer, E. T. Harley, 
        D. E. Fishkind, R. J. Vogelstein, and C. E. Priebe, “Fast approximate quadratic programming 
        for graph matching,” PLOS one, vol. 10, no. 4, p. e0121002, 2015.

    """

    def __init__(self, n_init, init_method):

        if n_init > 0 and type(n_init) is int:
            self.n_init = n_init
        else:
            msg = '"n_init" must be a positive integer'
            raise TypeError(msg)
        if init_method == "rand":
            self.init_method = "rand"
        elif init_method == "barycenter":
            self.init_method = "barycenter"
            n_init = 1
        else:
            msg = 'Invalid "init_method" parameter string'
            raise ValueError(msg)

    def fit(self, A, B):
        """
        Fits the model with two assigned adjacency matrices
        
        Parameters
        ---------
        A : 2d-array, square
            A square adjacency matrix
            
        B : 2d-array, square
            A square adjacency matrix
        
        Returns
        -------
        
        self : returns an instance of self
        """
        if A.shape[0] != B.shape[0]:
            msg = "Matrix entries must be of equal size"
            raise ValueError(msg)
        elif A.shape[0] != A.shape[1] or B.shape[0] != B.shape[1]:
            msg = "Matrix entries must be square"
            raise ValueError(msg)
        elif (A >= 0).all() == False or (B >= 0).all() == False:
            msg = "Matrix entries must be greater than or equal to zero"
            raise ValueError(msg)
        else:

            n = A.shape[0]  # number of vertices in graphs
            At = np.transpose(A)  # A transpose
            Bt = np.transpose(B)  # B transpose
            score = math.inf
            perm_inds = np.zeros(n)

            for i in range(self.n_init):

                # setting initialization matrix
                if self.init_method == "rand":
                    sk = SinkhornKnopp()
                    K = np.random.rand(
                        n, n
                    )  # generate a nxn matrix where each entry is a random integer [0,1]
                    for i in range(10):  # perform 10 iterations of Sinkhorn balancing
                        K = sk.fit(K)
                    J = np.ones((n, n)) / float(
                        n
                    )  # initialize J, a doubly stochastic barycenter
                    P = (K + J) / 2
                elif self.init_method == "barycenter":
                    P = np.ones((n, n)) / float(n)

                # OPTIMIZATION WHILE LOOP BEGINS
                for i in range(30):

                    delta_f = (
                        A @ P @ Bt + At @ P @ B
                    )  # computing the gradient of f(P) = -tr(APB^tP^t)
                    rows, cols = linear_sum_assignment(
                        delta_f
                    )  # run hungarian algorithm on gradient(f(P))
                    Q = np.zeros((n, n))
                    Q[rows, cols] = 1  # initialize search direction matrix Q

                    def f(x):  # computing the original optimization function
                        return np.trace(
                            At
                            @ (x * P + (1 - x) * Q)
                            @ B
                            @ np.transpose(x * P + (1 - x) * Q)
                        )

                    alpha = minimize_scalar(
                        f, bounds=(0, 1), method="bounded"
                    ).x  # computing the step size
                    P = alpha * P + (1 - alpha) * Q  # Update P
                # end of FW optimization loop

                row, perm_inds_new = linear_sum_assignment(
                    -P
                )  # Project onto the set of permutation matrices
                perm_mat_new = np.zeros((n, n))  # initiate a nxn matrix of zeros
                perm_mat_new[row, perm_inds_new] = 1  # set indices of permutation to 1
                score_new = np.trace(
                    np.transpose(A) @ perm_mat_new @ B @ np.transpose(perm_mat_new)
                )  # computing objective function value

                if score_new < score:  # minimizing
                    score = score_new
                    perm_inds = perm_inds_new

            self.perm_inds_ = perm_inds  # permutation indices
            self.score_ = score  # objective function value
            return self
