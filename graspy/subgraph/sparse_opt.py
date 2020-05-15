# Copyright 2020 NeuroData (http://neurodata.io)
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

import time
import numbers
import math

import numpy as np

from .base import BaseClassify


class SparseOpt(BaseClassify):
    """
    Network classification algorithm using sparse optimization.
    Fits a regularized logistic regression to a set of network
    adjacency matrices with responses, and returns an object
    with the classifier. The classifier fits a matrix of coefficients.

    Parameters
    ----------
    x : ndarray
        A matrix with the training samples, in which
        each row represents the vectorized (by column order)
        upper triangular part of a network adjacency matrix.

    y : ndarray, [0, 1]
        A vector containing the class labels of the training samples.

    xtest : ndarray
        Matrix containing the test samples, with
        each row representing an upper-triangular
        vectorized adjacency matrix.

    ytest : ndarray
        The labels of `xtest`.

    lambda_ : float, optional
        Penalty parameter, by default is 0.

    rho : float, optional
        Penalty parameter controlling sparsity, by default is 0.

    gamma : float, optional
        Ridge parameter (for numerical purposes), by default is 1e-5.

    Returns
    -------
    object
        Containing the trained graph classifier:
            - beta : ndarray
                Edge coefficients vector of the regularized
                logistic regression solution.

            - b : float
                Intercept value

            - yfit : ndarray
                Fitted logistic regression probabilities in the train data.

            - ypred : ndarray
                Predicted class for the test samples

            - train_error : float
                Percentage of train samples that are misclassified.

            - test_error : float
                Percentage of test samples that are misclassified.

            - active_nodes : ndarray
                Estimated subgraph nodes.

    References
    ----------
    .. [1] J. Relion, et al. "Network Classification with
        Applications to Brain Connectomics," arXiv: 1701.08140 [stat.ME], 2019
    """

    def __init__(self, lambda_=0, gamma=1e-5, rho=0, nodes=None):
        if not isinstance(lambda_, numbers.Real):
            raise ValueError("lambda must be float or int")
        if lambda_ < 0:
            raise ValueError("lambda must be a positive value")

        if not isinstance(gamma, numbers.Real):
            raise ValueError("gamma must be float or int")

        if not isinstance(rho, numbers.Real):
            raise ValueError("rho must be float or int")
        if rho < 0 or rho > 1:
            raise ValueError("rho must be between 0 and 1")

        self.lambda_ = lambda_
        self.gamma = gamma
        self.rho = rho
        self.nodes = nodes

    def _admm(self, u_in, rho, omega1, omega2, beta_start=None):
        tol=1e-7
        MAX_ITER = 200
        
        n = u_in.size
        m = self.D.shape[0]

        if beta_start is None:
            beta = u_in
        else:
            beta = beta_start

        soft_beta = _soft_threshold(u_in, omega1)
        q = soft_beta
        r = self.D @ q

        if np.max(np.absolute(soft_beta)) == 0:
            beta = soft_beta
            i = 0
            conv_crit = 0
            best_beta = soft_beta

            self.beta = beta
            self.q = q
            self.r = r
            self.iter = i
            self.conv_crit = conv_crit
            self.best_beta = best_beta

            return beta, q, r, i, conv_crit, best_beta

        qk = q
        rk = r
        u = np.zeros((n, 1))
        v = np.zeros((m, 1))
        i = 1
        phi_beta_k = (
            0.5 * np.linalg.norm(u_in - beta) ** 2
            + omega1 * np.sum(np.abs(beta))
            + omega2 * _gl_penalty(self, self.D @ beta)
        )
        
        conv_crit = np.infty
        sk = np.infty
        resk = np.infty
        best_beta = beta
        best_phi = phi_beta_k

        while (resk > tol or sk > tol) and i <= MAX_ITER:
            aux = u_in - u + (rho * q) + _cross(self.D, rho * r - v)
            beta = aux / (1 + 3 * rho)
            Dbeta = self.D @ beta

            # update q
            q = _soft_threshold(beta + u / rho, omega1 / rho)

            # update r
            Dbetavrho = Dbeta + v / rho
            r = _soft_threshold_12(Dbetavrho, omega2 / rho)

            u += rho * (beta - q)
            v += rho * (Dbeta - r)

            # update convergence criteria
            phi_beta_k1 = (
                0.5 * (np.linalg.norm(beta - u_in)) ** 2
                + omega1 * np.sum(np.abs(beta))
                + omega2 * _gl_penalty(self, Dbeta)
            )

            sk = rho * np.max(np.absolute(q - qk)) + np.max(
                np.absolute(_cross(self.D, r - rk))
            )

            res1k = np.linalg.norm(beta - q)
            res2k = np.linalg.norm(Dbeta - r)

            resk = res1k + res2k
            qk = q
            rk = r
            conv_crit = np.abs(phi_beta_k1 - phi_beta_k) / phi_beta_k
            phi_beta_k = phi_beta_k1
            
            if phi_beta_k1 < best_phi:
                best_beta = beta
                best_phi = phi_beta_k
                break

            i += 1

        beta_q = beta 
        beta_q[np.where(q == 0)] = 0
        
        phi_beta_q = (
            0.5 * (np.linalg.norm(u_in - beta_q)) ** 2
            + omega1 * np.sum(np.abs(beta_q))
            + omega2 * _gl_penalty(self, self.D @ beta_q)
        )

        whichm = np.min([phi_beta_k1, best_phi, phi_beta_q])
        if whichm == 1:
            best_beta = beta
        elif whichm == 3:
            best_beta = beta_q

        self.beta = beta
        self.q = q
        self.r = r
        self.iter = i
        self.conv_crit = conv_crit
        self.best_beta = best_beta
        self.best_phi = best_phi

        return beta, q, r, i, conv_crit, best_beta

    def _logistic_lasso(self, lambda1, lambda2):
        rho = 1
        n = self.y.size
        p = self.x.shape[1]

        def b_derivative(Xbeta, b):
            return np.sum(-self.y / (1 + np.exp(self.y * (Xbeta + b)))) / n

        def b_hessian(Xbeta, b):
            return (
                np.sum(
                    1
                    / (np.exp(
                        -self.y * (Xbeta + b)
                    ) + np.exp(self.y * (Xbeta + b)) + 2)
                )
                / n
            )

        def grad_f(Xbeta, b, beta):
            return (
                -_cross(
                    self.x, self.y / (1 + np.exp(self.y * (Xbeta + b)))
                ) / n
                + self.gamma * beta
            )

        def f(Xbeta, b, beta):
            return np.sum(np.log(1 + np.exp(-self.y * (Xbeta + b)))) / n
            +self.gamma / 2 * _cross(beta, beta)

        def penalty(beta):
            return lambda1 * np.sum(np.abs(beta))
            +lambda2 * _gl_penalty(self, self.D @ beta)

        def proximal(u, lambda_, beta_startprox=None, tol=1e-7):
            if lambda2 > 0:
                gl = self._admm(
                    u,
                    rho,
                    omega1=lambda_ * lambda1,
                    omega2=lambda_ * lambda2,
                    beta_start=beta_startprox,
                )
                return gl[5], gl[1], gl[2]
            elif lambda1 > 0:
                val = np.abs(u) - (lambda1 * lambda_)
                return (np.sign(u) * val), lambda1
            else:
                return u, lambda1

        def b_step(xbeta, b_start=0):
            tolb = 1e-4
            max_sb = 100
            b_n = b_start
            i = 0
            b_deriv = np.inf
            while np.abs(b_deriv) > tolb and i < max_sb:
                b_deriv = b_derivative(xbeta, b_n)
                b_hess = b_hessian(xbeta, b_n)
                b_n -= b_deriv / (b_hess + (np.abs(b_deriv / b_hess) > 100))
                i += 1
            return b_n

        beta_start = np.zeros((p, 1))
        b_start = 0

        optimal = self._fista(
            proximal, b_step, f, grad_f, penalty, beta_start, b_start
        )

        return optimal

    def _fista(
        self, proximal, b_step, f, grad_f, penalty, beta_start, b_start
    ):
        TOLERANCE = self.conv_crit
        MAX_STEPS = 300
        MAX_TIME = self.max_time

        x_start = beta_start
        xk1 = x_start
        xk = x_start
        criterion = np.infty
        crit_f = np.infty

        k = 0
        tk = 0.125
        beta_step = 0.5

        xbeta = self.x @ x_start
        b = b_step(xbeta, b_start)
        
        best_beta = x_start
        
        best_b = b

        best_f = f(xbeta, b, x_start) + penalty(x_start)

        newf = best_f

        time_start = time.time()

        crit_f_1 = crit_f

        beta_path = []

        beta_path.append(xk1)

        while (k <= 5) or (
            (crit_f > TOLERANCE and criterion > TOLERANCE)
            and (k < MAX_STEPS)
            and (int(time.time() - time_start) < MAX_TIME)
        ):
            crit_f = newf
            is_best_end = False
            
            k += 1
            xk_1 = xk
            xk = xk1
            y = xk + ((k - 2) / (k + 1)) * (xk - xk_1)

            boolean = True

            while boolean:
                Xy = self.x @ y
                z = y - tk * grad_f(Xy, b, y)
                prox = proximal(z, tk, y, tol=(1e-2 / k))
                z = prox[0]
                xbeta = self.x @ z
                if f(xbeta, b, z) <= (
                    f(Xy, b, y)
                    + np.transpose(grad_f(Xy, b, y)) @ (z - y)
                    + 1 / (2 * tk) * (np.linalg.norm(z - y)) ** 2
                ):
                    break
                tk *= beta_step

            b = b_step(xbeta, b)
            xk1 = z
            criterion = np.linalg.norm(xk1 - xk)
            newf = f(xbeta, b, z) + penalty(xk1)
            if crit_f > 0:
                crit_f = np.absolute(newf - crit_f) / crit_f
            if newf < best_f:
                is_best_end = True
                self.best_beta = xk1
                self.best_b = b
                self.best_prox = prox
                self.best_f = newf
            if (crit_f == 0) or (
                (np.absolute(crit_f - crit_f_1) / crit_f) < 0.1
            ):
                tk *= 2
            crit_f_1 = crit_f
            beta_path.append(xk1)

        self.beta_path = beta_path

        optimal = [xk1, xk, criterion, crit_f, k, tk, beta_step, b]

        return optimal

    def _predict(self):
        pass

    def fit(self, x, y, xtest, ytest):
        self.nodes = int((1 + np.sqrt(1 + 8 * x.shape[1])) / 2)

        # Error Checking
        if not isinstance(self.nodes, int):
            raise ValueError("nodes must be int")

        if type(x) is not np.ndarray:
            raise TypeError("x must be numpy.ndarray")
        if type(y) is not np.ndarray:
            raise TypeError("y must be numpy.ndarray")
        if type(xtest) is not np.ndarray:
            raise TypeError("xtest must be numpy.ndarray")
        if type(ytest) is not np.ndarray:
            raise TypeError("ytest must be numpy.ndarray")

        if len(x.shape) != 2:
            raise ValueError("x must be a matrix")
        if len(xtest.shape) != 2:
            raise ValueError("xtest must be a matrix")
        if len(y.shape) != (2 or 1):
            raise ValueError("y must be a vector")
        if len(ytest.shape) != (2 or 1):
            raise ValueError("ytest must be a vector")

        if y.shape != (x.shape[0], 1):
            raise ValueError("y must have shape (x.size, 1)")
        if ytest.shape != (xtest.shape[0], 1):
            raise ValueError("ytest must have shape (xtest.size, 1)")

        # Start of code
        y = y.astype(float)
        self.y = 2 * ((y - np.min(y)) / (np.max(y) - np.min(y))) - 1

        self.Ypos_label = 1
        self.Yneg_label = 0

        # np.float64 makes std more accurate
        alpha_norm = np.max(np.std(x, dtype=np.float64, axis=1, ddof=1))
        self.x = x / alpha_norm
        lambda1 = self.lambda_ * self.rho / alpha_norm
        lambda2 = self.lambda_ / alpha_norm
        self.gamma = self.gamma / alpha_norm

        # Create D
        self.D = _construct_d(self.nodes)

        self.beta_start = np.zeros((int(self.nodes * (self.nodes - 1) / 2), 1))
        self.b_start = 0
        self.max_iter = 300
        self.conv_crit = 1e-5
        self.max_time = np.infty

        # Run logistic
        self._logistic_lasso(lambda1, lambda2)
        self.beta = self.best_beta / alpha_norm
        self.b = self.best_b

        yfit = alpha_norm * (self.x @ self.beta) + self.b
        self.yfit = np.exp(yfit) / (1 + np.exp(yfit))
        self.train_error = 1 - np.sum(np.diag(yfit)) / len(self.y)

        self._predict()

        pred = xtest @ self.beta + self.b
        Ypred = pred
        Ypred[Ypred > 0] = self.Ypos_label
        Ypred[Ypred <= 0] = self.Yneg_label

        self.ypred = Ypred

        self.test_error = np.sum(ytest != self.ypred) / self.ypred.size

        adj_inds = np.triu_indices(self.nodes, 1)
        ind_val = x.shape[1]

        Adj_matr = np.zeros((self.nodes, self.nodes))
        Adj_matr[adj_inds] = self.beta.reshape(ind_val)
        Adj_matr += np.transpose(Adj_matr)
        subgraph = np.where(np.sum(Adj_matr, 1) != 0)[0]
        subgraph = subgraph.reshape((self.nodes, 1))
        self.adj_matr = Adj_matr
        self.active_nodes = subgraph

        self.yfit_test = (xtest @ self.beta) + self.b

        return self


def _soft_threshold(x, lambda_):
    n = x.shape[0]

    for i in range(n):
        if x[i] > lambda_:
            x[i] -= lambda_
        elif x[i] < -lambda_:
            x[i] += lambda_
        else:
            x[i] = 0

    return x


def _soft_threshold_12(x, lambda_):
    n = x.shape[0]
    m = x.shape[1]
    norm = np.linalg.norm(x)
    
    if norm <= lambda_:
        x = np.zeros((n, m))
    else:
        t = (1 - lambda_ / norm)
        x *= t
    
    return x
        
def _gl_penalty(self, b):
    gl = 0
    
    for i in range(self.nodes):
        norm_node = 0
        j = (self.nodes - 1) * i
        
        while j < (self.nodes - 1) * (i + 1):
            norm_node += b[j][0] * b[j][0]
            j += 1
        
        gl += math.sqrt(norm_node)

    return gl


def _construct_d(nodes=264):
    m = nodes * (nodes - 1)
    n = int(nodes * (nodes - 1) / 2)
    D = np.zeros((m, n))

    B = np.zeros((nodes, nodes))

    for j in range(0, nodes - 1):
        row = j
        for k in range((j + 1), nodes):
            if row == j:
                B[j, k] = int((j + 1) * (j + 2) / 2)
                row += 1
            else:
                B[j, k] = B[j, k - 1] + row
                row += 1

    B = np.transpose(B) + B

    indexer = B[np.nonzero(B)].astype(int)

    D = np.zeros(shape=(nodes * (nodes - 1), int(nodes * (nodes - 1) / 2)))

    count = 0
    for i in range(0, nodes):
        for a in range(0, nodes - 1):
            D[
                (i * (nodes - 1)):((i + 1) * (nodes - 1))
            ][a][indexer[count] - 1] = 1
            count += 1

    return D


def _cross(v1, v2):

    out = v1.transpose() @ v2

    m = v1.shape[1]
    n = v2.shape[1]

    return out.reshape((m, n))
