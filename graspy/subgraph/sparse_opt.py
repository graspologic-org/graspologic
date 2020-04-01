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

import math
import time

import numpy as np

from .base2 import BaseClassify


class SparseOpt(BaseClassify):
    """
    Network classification algorithm using sparse optimization.
    Fits a regularized logistic regression to a set of network adjacency matrices with responses, and returns an
    object with the classifier.The classifier fits a matrix of coefficients.
    Parameters
    ----------
    x : ndarray
        A matrix with the training samples, in which each row represents the vectorized (by column order) upper
        triangular part of a network adjacency matrix.
    y : ndarray, [0, 1]
        A vector containing the class labels of the training samples.
    xtest : ndarray
        Matrix containing the test samples, with each row representing an upper-triangular vectorized adjacency
        matrix.
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
                Edge coefficients vector of the regularized logistic regression solution.
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
            - active_nodes : ndarray (added this)
                Estimated subgraph.
    References
    ----------
    .. [1] J. Relion, D. Kessler, E. Levina, S. Taylor.  "Network Classification
        with applications to brain connectomics," arXiv: 1701.08140 [stat.ME], 2019
    """
    #Check with TAs about if this satisfies graspy api, what is necessary for users
    def __init__(self, lambda_=0, gamma=1e-5, rho=0, nodes=None):
        self.lambda_ = lambda_
        self.gamma = gamma
        self.rho = rho #added this
        self.nodes = nodes
    
    def _admm(self, omega1, omega2, tol=1e-7, beta_start=None):
        self.rho = 1
        
        n = self.y.size
        m = self.D.shape[0]

        if not beta_start:
            beta = self.y
        else:
            beta = beta_start

        soft_beta = _soft_threshold(self.y, omega1)
        q = soft_beta
        r = self.D @ q
        if np.max(np.absolute(q)) == 0:
            beta = q
            i = 0
            conv_crit = 0
            best_beta = q

            self.beta = beta
            self.q = q
            self.r = r
            self.iter = i
            self.conv_crit = conv_crit
            self.best_beta = best_beta

            return beta, q, r, i, conv_crit, best_beta

        qk = q
        rk = r
        u = np.zeros(n)
        v = np.zeros(m)
        i = 1
        phi_beta_k = (0.5*np.sum(self.y - beta @ self.y - beta) + omega1*np.sum(np.abs(beta)) #changed so inside of np.sum is squared
                      + omega2*_gl_penalty(self.D @ beta))
        conv_crit = np.infty
        sk = np.infty
        resk = np.infty
        best_beta = beta
        best_phi = phi_beta_k

        #no is_best_end included anywhere
        
        while (resk > tol or sk > tol) and i <= self.max_iter:
            aux = self.y-u + self.rho*q - self.D @ (self.rho*r - v) #is this a plus?
            beta = aux / (1+3*self.rho)
            Dbeta = self.D @ beta

            # update q
            q = _soft_threshold(beta+u/self.rho, omega1/self.rho)

            # update r
            Dbetavrho = Dbeta + v/self.rho
            r = _soft_threshold(Dbetavrho, omega2/self.rho)

            u = u + self.rho * (beta - q)
            v = v + self.rho * (Dbeta - r)

            # update convergence criteria
            phi_beta_k1 = (0.5*np.sum(self.y - beta @ self.y - beta) #changed so inside of np.sum is squared
                           + omega1*np.sum(np.abs(beta))
                           + omega2*_gl_penalty(self.D, self.nodes))
            
            sk = (self.rho * np.max(np.absolute(q - qk))
                  + np.max(np.absolute(np.cross(self.D, r-rk))))

            res1k = np.sqrt(np.sum(beta - q))
            res2k = np.sqrt(np.sum(Dbeta - r))

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

        beta_q = beta * (q == 0).astype(int)
        phi_beta_q = (0.5*np.sum(self.y - beta_q @ self.y - beta_q) #changed so inside of np.sum is squared
                      + omega1*np.sum(np.abs(beta_q))
                      + omega2*_gl_penalty(self.D @ beta_q))

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

        return beta, q, r, i, conv_crit, best_beta

    def _logistic_lasso(self, lambda1, lambda2):
        self.rho = 1
        n = self.y.size
        p = self.x.shape[1]
        
        b_derivative = lambda xbeta, b: (np.sum(-self.y / (1 +
            np.exp(self.y * (xbeta+b)))) / n)
        
        b_hessian = lambda xbeta, b: (np.sum(1 / (np.exp(-self.y * (xbeta+b))
            + np.exp(self.y * (xbeta+b)) + 2)) / n)

        grad_f = lambda xbeta, b, beta: (-np.cross(self.x,
            self.y / (1 + np.exp(self.y * (xbeta+b)))) / n
            + self.gamma*self.beta)

        f = lambda xbeta, b, beta: (np.sum(np.log(1 + np.exp(-self.y
            * (xbeta+b)))) / n + self.gamma/2 * np.cross(beta, beta))
        penalty = lambda beta: (lambda1*np.sum(np.abs(beta))
            + lambda2*_gl_penalty(self.D@beta, self.nodes))

        def proximal(u, lambda_, beta_startprox=None, tol=1e-7):
            if lambda2 > 0:
                gl = self._admm(omega1=lambda_*lambda1,
                    omega2=lambda_*lambda2, beta_start=beta_startprox)
                return gl[5], gl[1], gl[2]
            elif lambda1 > 0:
                return np.sign(np.max(np.abs(u) - (lambda1*lambda_), 0))
            else:
                return u

        def b_step(xbeta, b_start=0):
            tolb = 1e-4
            max_sb = 100
            b_n = b_start
            i = 0
            b_deriv = np.inf
            while np.abs(b_deriv) > tolb and i < max_sb: #possibly change this back to b_derivative ?
                b_deriv = b_derivative(xbeta, b_n) #possibly change this back to b_deriv ?
                b_hess = b_hessian(xbeta, b_n)
                b_n -= b_deriv/(b_hess + 1 * (np.abs(b_deriv/b_hess) > 100)) #change back to = from -=?
                i += 1
            return b_n

        beta_start = np.zeros(p)
        b_start = 0

        optimal = self._fista(proximal, b_step, f, grad_f, penalty,
                              beta_start, b_start)

        return optimal

    def _fista(self, proximal, b_step, f, grad_f, penalty, beta_start,
               b_start): #Finish up this
        TOLERANCE = 1e-6 #added
        MAX_STEPS = 300 #added
        MAX_TIME = 10800
        
        x_start = beta_start #added
        xk1 = x_start
        xk = x_start
        criterion = np.infty
        crit_f = np.infty

        k = 0
        tk = 0.125
        beta_step = 0.5

        xbeta = x @ x_start
        b = b_step(xbeta, b_start)
        best_beta = x_start
        best_b = b
        
        best_f = f(xbeta, b, x_start) + penalty(x_start, b)
        
        newf = best_f
        
        time_start = time.time()
        
        crit_f_1 = crit_f
        
        beta_path = []
        
        beta_path.append(xk1)
        
        is_best_end = False
        
        while (k <= 5) or ((crit_f > TOLERANCE & criterion > TOLERANCE) & (k < MAX_STEPS) & (time.time() - time_start < MAX_TIME)):
            crif_f = newf
            is_best_end = False
            k += 1
            xk_1 = xk
            xk = xk1
            y = xk + ((k - 2) / (k + 1)) * (xk - xk_1)
            
            boolean = True
            
            while boolean:
                Xy = self.x @ y
                z = y - tk * grad_f(Xy, b, y)
                prox = proximal(z, tk, y, tol=(1e-2/k))
                z = prox[0]
                xbeta = x @ z
                if f(xbeta, b, z) <= (f(Xy, b, y).astype(double) + np.transpose(grad_f(Xy, b ,y)) @ (z - y) + 1 / (2 * tk) * np.sum((z - y) @ (z - y))):
                    break
                tk *= beta_step
                
            b = b_step(xbeta, b)
            xk1 = z
            criterion = np.sum((xk1 - xk) @ (xk1 - xk))
            newf = f(xbeta, b, z) + penalty(xk1, b)
            if (crit_f > 0):
                crit_f = np.absolute(newf - crit_f) / crit_f
            if (newf < best_f):
                is_best_end = True
                self.best_beta = xk1
                self.best_b = b
                self.best_prox = prox
                self.best_f = newf
            if (crit_f == 0) or ((np.absolute(crit_f - crit_f_1) / crit_f) < 0.1):
                tk *= 2
            crit_f_1 = crit_f
            beta_path.append(xk1)
        
        self.beta_path = beta_path
        
        optimal = [xk1, xk, criterion, crit_f, k, tk, beta_step, b]
        
        return optimal

    def _predict(self):
        pass

    def fit(self, x, y, xtest):
        self.nodes = int((1 + np.sqrt(1 + 8*x.shape[1])) / 2)
        print(self.nodes)
        yorig = y
        self.y = 2 * (y - np.min(y)) / (np.max(y) - np.min(y)) - 1

        alpha_norm = np.std(x)
        self.x = x / alpha_norm
        lambda1 = self.lambda_ * self.rho / alpha_norm
        lambda2 = self.lambda_ / alpha_norm
        self.gamma = self.gamma / alpha_norm #added / alpha_norm and changed to self.gamma on the left
        
        self.D = _construct_d(self.nodes)

        self.beta_start = np.zeros(self.nodes*(self.nodes-1)/2)
        self.b_start = 0
        self.max_iter = 200
        self.conv_crit = 1e-5 #possibly change back to 1e-6
        self.max_time = np.infty

        self._logistic_lasso(lambda1, lambda2)
        self.beta = self.best_beta / alpha_norm
        self.b = self.best_b
        
        yfit = alpha_norm * (self.x @ self.beta) + self.b #changed from beta to self.beta, added + self.b
        self.yfit = np.exp(yfit) / (1 + np.exp(yfit))
        self.train_error = 1 - np.sum(np.diag(yfit)) / len(self.y)
        
        self._predict()
        yfit_test = xtest @ self.beta + self.b #changed to self.beta and self.b
        # Compare with vertex screening accuracy later
        Adj_matr = np.zeros((self.nodes, self.nodes))
        Adj_matr[np.triu_indices(self.nodes, self.nodes)] = self.beta
        Adj_matr += np.transpose(Adj_matr)
        np.fill_diagonal(Adj_matr, 0)
        subgraph = np.where(np.sum(Adj_matr, 1) != 0)
        self.active_nodes = subgraph
        
        return self


def _soft_threshold(x, lambda_, nodes=None):
    n = x.shape[0]

    for i in range(n):
        norm_node = _gl_penalty(x, nodes)
        t = 1 - lambda_/norm_node
        for j in range((nodes-1)*i, (nodes-1)*(i+1)):
            if norm_node <= lambda_:
                x[j] = 0
            else:
                x[j] *= t

    return x

def _gl_penalty(b, nodes=None):
    for i in range(nodes):
        norm_node = np.sum([b[j] ** 2 for j in range((nodes-1)*i, (nodes-1)*(i+1))])
        gl = np.sqrt(norm_node)

    return gl

def _construct_d(nodes):
    B = np.zeros((nodes, nodes))
    print(B[np.triu_indices(nodes, k=1)].shape)
    B[np.triu_indices(nodes, k=1)] = np.asarray(range(1, int(nodes * (nodes-1) / 2) + 1))
    B = B + B.T
    D = np.zeros(shape=(nodes*(nodes-1), int(nodes * (nodes-1) / 2))

    for i in range(nodes):
        D[i*(nodes-1) + 1:(i+1)*(nodes-1) + 1, B[i, -i]] = np.diag(nodes-1)

    return D