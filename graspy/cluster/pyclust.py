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
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture.gaussian_mixture import _estimate_gaussian_parameters
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
from sklearn.model_selection import ParameterGrid

from .base import BaseCluster

class PyclustCluster(BaseCluster):
    """
    Pyclust Cluster.

    Clustering algorithm using a hierarchical agglomerative clustering then Gaussian
    mixtured model (GMM) fitting. Different combinations of agglomeration, GMM, and 
    cluster numbers are used and the clustering with the best Bayesian Information
    Criterion (BIC) is chosen.


    Parameters
    ----------
    min_components : int, default=2. 
        The minimum number of mixture components to consider (unless
        max_components=None, in which case this is the maximum number of
        components to consider). If max_componens is not None, min_components
        must be less than or equal to max_components.

    max_components : int or None, default=None.
        The maximum number of mixture components to consider. Must be greater 
        than or equal to min_components.

    affinity : {'euclidean','manhattan','cosine','none', 'all' (default)}, optional
        String or list/array describing the type of affinities to use in agglomeration.
        If a string, it must be one of:

        - 'euclidean'
            L2 norm
        - 'manhattan'
            L1 norm
        - 'cosine'
            cosine similarity
        - 'none'
            no agglomeration - GMM is initialized with k-means
        - 'all'
            considers all affinities in ['euclidean','manhattan','cosine','none']
        If a list/array, it must be a list/array of strings containing only
            'euclidean', 'manhattan', 'cosine', and/or 'none'.

    linkage : {'ward','complete','average','single', 'all' (default)}, optional
        String or list/array describing the type of linkages to use in agglomeration.
        If a string, it must be one of:

        - 'ward'
            ward's clustering, can only be used with euclidean affinity
        - 'complete'
            complete linkage
        - 'average'
            average linkage
        - 'single'
            single linkage
        - 'all'
           considers all linkages in ['ward','complete','average','single']
        If a list/array, it must be a list/array of strings containing only
            'ward', 'complete', 'average', and/or 'single'.
        
    covariance_type : {'full', 'tied', 'diag', 'spherical', 'all' (default)} , optional
        String or list/array describing the type of covariance parameters to use.
        If a string, it must be one of:
        
        - 'full'
            each component has its own general covariance matrix
        - 'tied'
            all components share the same general covariance matrix
        - 'diag'
            each component has its own diagonal covariance matrix
        - 'spherical'
            each component has its own single variance
        - 'all'
            considers all covariance structures in ['spherical', 'diag', 'tied', 'full']
        If a list/array, it must be a list/array of strings containing only
            'spherical', 'tied', 'diag', and/or 'spherical'.
    
    random_state : int, RandomState instance or None, optional (default=None)
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by ``np.random``.

    Attributes
    ----------
    results_ : pandas.DataFrame
        Contains exhaustive information about all the clustering runs.
        Columns are:
            'model' - fit GaussianMixture object
            'bic' - Bayesian Information Criterion
            'ari' - Adjusted Rand Index, nan if y is not given
            'n_components' - number of clusters
            'affinity' - affinity used in Agglomerative Clustering
            'linkage' - linkage used in Agglomerative Clustering
            'covariance_type' - covariance type used in GMM
            'reg_covar' : regularization used in GMM

    bic_ : the best (lowest) Bayesian Information Criterion
    n_components_ : number of clusters in the model with the best BIC
    covariance_type_ : covariance type in the model with the best BIC
    affinity_ : affinity used in the model with the best BIC
    linkage_ : linkage used in the model with the best BIC
    reg_covar_ : regularization used in the model with the best BIC
    ari_ : ARI from the model with the best BIC, nan if no y is given
    model_ : GaussianMixture object with the best BIC
    """

    def __init__(
        self,
        min_components=2,
        max_components=None,
        affinity = "all",
        linkage = "all",
        covariance_type="all",
        random_state=None,
    ):
        if isinstance(min_components, int):
            if min_components <= 0:
                msg = "min_components must be >= 1."
                raise ValueError(msg)
        else:
            msg = "min_components must be an integer, not {}.".format(
                type(min_components)
            )
            raise TypeError(msg)

        if isinstance(max_components, int):
            if max_components <= 0:
                msg = "max_components must be >= 1 or None."
                raise ValueError(msg)
            elif min_components > max_components:
                msg = "min_components must be less than or equal to max_components."
                raise ValueError(msg)
        elif max_components is not None:
            msg = "max_components must be an integer or None, not {}.".format(
                type(max_components)
            )
            raise TypeError(msg)

        if isinstance(affinity, (np.ndarray, list)):
            affinity = np.unique(affinity)
        elif isinstance(affinity, str):
            if affinity == "all":
                affinity = ["euclidean","manhattan","cosine","none"]
            else:
                affinity = [affinity]
        else:
            msg = "affinity must be a numpy array, a list, or "
            msg += "string, not {}".format(type(affinity))
            raise TypeError(msg)

        for aff in affinity:
            if aff not in ["euclidean","manhattan","cosine","none"]:
                msg = (
                    "affinity must be one of "
                    + '["euclidean","manhattan","cosine","none"]'
                )
                msg += " not {}".format(aff)
                raise ValueError(msg)

        if isinstance(linkage, (np.ndarray, list)):
            linkage = np.unique(linkage)
        elif isinstance(linkage, str):
            if linkage == "all":
                linkage = ["ward", "complete", "average", "single"]
            else:
                linkage = [linkage]
        else:
            msg = "linkage must be a numpy array, a list, or "
            msg += "string, not {}".format(type(linkage))
            raise TypeError(msg)

        for link in linkage:
            if link not in ["ward", "complete", "average", "single"]:
                msg = (
                    "covariance structure must be one of "
                    + '["ward", "complete", "average", "single"]'
                )
                msg += " not {}".format(link)
                raise ValueError(msg)

        if isinstance(covariance_type, (np.ndarray, list)):
            covariance_type = np.unique(covariance_type)
        elif isinstance(covariance_type, str):
            if covariance_type == "all":
                covariance_type = ["spherical", "diag", "tied", "full"]
            else:
                covariance_type = [covariance_type]
        else:
            msg = "covariance_type must be a numpy array, a list, or "
            msg += "string, not {}".format(type(covariance_type))
            raise TypeError(msg)

        for cov in covariance_type:
            if cov not in ["spherical", "diag", "tied", "full"]:
                msg = (
                    "covariance structure must be one of "
                    + '["spherical", "diag", "tied", "full"]'
                )
                msg += " not {}".format(cov)
                raise ValueError(msg)

        new_covariance_type = []
        for cov in ["spherical", "diag", "tied", "full"]:
            if cov in covariance_type:
                new_covariance_type.append(cov)

        self.min_components = min_components
        self.max_components = max_components
        self.affinity = affinity
        self.linkage = linkage
        self.covariance_type = new_covariance_type
        self.random_state = random_state

    def _process_paramgrid(self, paramgrid):
        """
        Removes combinations of affinity and linkage that are not possible.

        Parameters
        ----------
        paramgrid : list of dicts
            Each dict has the keys 'affinity', 'covariance_type', 'linkage',
            'n_components', and 'random_state'

        Returns
        -------
        paramgrid_processed : list pairs of dicts
            For each pair, the first dict are the options for AgglomerativeClustering.
            The second dict include the options for GaussianMixture.
        """
        paramgrid_processed = []
        for params in paramgrid:
            if params['affinity'] == 'none' and params['linkage'] != 'ward':
                pass
            elif params['linkage'] == 'ward' and params['affinity'] != 'euclidean':
                pass
            else:
                gm_keys = ['covariance_type', 'n_components', 'random_state']
                gm_params = {key:params[key] for key in gm_keys}

                ag_keys = ['affinity', 'linkage']
                ag_params = {key:params[key] for key in ag_keys}
                ag_params['n_clusters'] = params['n_components']

                paramgrid_processed.append([ag_params, gm_params])
        
        return paramgrid_processed

    def _labels_to_onehot(self, labels):
        """
        Converts labels to one-hot format.

        Parameters
        ----------
        labels : ndarray, shape (n_samples,)
            Cluster labels

        Returns
        -------
        onehot : ndarray, shape (n_samples, n_clusters)
            Each row has a single one indicating cluster membership.
            All other entries are zero.
        """
        n = len(labels)
        k = max(labels)+1
        onehot = np.zeros([n,k])
        onehot[np.arange(n),labels] = 1
        return onehot
    
    def  _onehot_to_initialparams(self, X, onehot, cov_type):
        """
        Computes cluster weigts, cluster means and cluster precisions from
        a given clustering.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        onehot : ndarray, shape (n_samples, n_clusters)
            Each row has a 1 indicating cluster membership, other entries are 0.
        cov_type : {'full', 'tied', 'diag', 'spherical'}
            Covariance type for Gaussian mixture model
        """
        n = X.shape[0]
        weights, means, covariances = _estimate_gaussian_parameters(
            X, onehot, 1e-06, cov_type)
        weights /= n

        precisions_cholesky_ = _compute_precision_cholesky(
            covariances, cov_type)

        if cov_type=="tied":
            c = precisions_cholesky_
            precisions = np.dot(c,c.T)
        elif cov_type=="diag":
            precisions = precisions_cholesky_
        else:
            precisions = [np.dot(c,c.T) for c in precisions_cholesky_]

        return weights, means, precisions
    
    def _increase_reg(self, reg):
        """
        Increase regularization factor by factor of 10.

        Parameters
        ----------
        reg: float
            Current regularization factor.

        Returns
        -------
        reg : float
            Increased regularization
        """
        if reg == 0:
            reg = 1e-06
        else:
            reg = reg*10
        return reg

    def fit(self, X, y=None):
        """
        Fits gaussian mixture model to the data.
        Initialize with agglomerative clustering then
        estimate model parameters with EM algorithm.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            List of n_features-dimensional data points. Each row
            corresponds to a single data point.
        
        y : array-like, shape (n_samples,), optional (default=None)
            List of labels for X if available. Used to compute
            ARI scores.

        Returns
        -------
        self
        """

         # Deal with number of clusters
        if self.max_components is None:
            lower_ncomponents = 1
            upper_ncomponents = self.min_components
        else:
            lower_ncomponents = self.min_components
            upper_ncomponents = self.max_components

        if upper_ncomponents > X.shape[0]:
            if self.max_components is None:
                msg = "if max_components is None then min_components must be >= "
                msg += "n_samples, but min_components = {}, n_samples = {}".format(
                    upper_ncomponents, X.shape[0]
                )
            else:
                msg = "max_components must be >= n_samples, but max_components = "
                msg += "{}, n_samples = {}".format(upper_ncomponents, X.shape[0])
            raise ValueError(msg)
        elif lower_ncomponents > X.shape[0]:
            msg = "min_components must be <= n_samples, but min_components = "
            msg += "{}, n_samples = {}".format(upper_ncomponents, X.shape[0])
            raise ValueError(msg)

        # Get parameters
        random_state = self.random_state

        param_grid = dict(
            affinity=self.affinity,
            linkage = self.linkage,
            covariance_type=self.covariance_type,
            n_components=range(lower_ncomponents, upper_ncomponents + 1),
            random_state=[random_state],
        )

        param_grid = list(ParameterGrid(param_grid))
        param_grid = self._process_paramgrid(param_grid)


        results = pd.DataFrame(columns=['model','bic','ari','n_components','affinity','linkage','covariance_type','reg_covar'])
        
        for params in param_grid:
            if params[0]['affinity'] != 'none':
                agg = AgglomerativeClustering(**params[0])
                agg_clustering = agg.fit_predict(X)
                onehot = self._labels_to_onehot(agg_clustering)
                weights_init, means_init, precisions_init = self._onehot_to_initialparams(
                    X, onehot, params[1]['covariance_type'])
                gm_params = params[1]
                gm_params['weights_init'] = weights_init
                gm_params['means_init'] = means_init
                gm_params['precisions_init'] = precisions_init
                gm_params['reg_covar'] = 0
            else:
                gm_params = params[1]
                gm_params['init_params'] = 'kmeans'
                gm_params['reg_covar'] = 1e-6

            bic = np.inf #if none of the iterations converge, bic is set to inf
            #below is the regularization scheme
            while gm_params['reg_covar'] <= 1:
                model = GaussianMixture(**gm_params)
                try:
                    model.fit(X)
                    predictions = model.predict(X)
                    counts = [sum(predictions == i) for i in range(gm_params['n_components'])]
                    #singleton clusters not allowed
                    assert not any([count <= 1 for count in counts])

                except ValueError:
                    gm_params['reg_covar'] = self._increase_reg(gm_params['reg_covar'])
                    continue
                except AssertionError:
                    gm_params['reg_covar'] = self._increase_reg(gm_params['reg_covar'])
                    continue
                #if the code gets here, then the model has been fit with no errors or singleton clusters
                bic = model.bic(X)
                break

            if y is not None:
                predictions = model.predict(X)
                ari = adjusted_rand_score(y, predictions)
            else:
                ari = float('nan')
            entry = pd.DataFrame({'model':[model],'bic':[bic],'ari':[ari],
                'n_components':[gm_params['n_components']],
                'affinity':[params[0]['affinity']],'linkage':[params[0]['linkage']],
                'covariance_type':[gm_params['covariance_type']],
                'reg_covar':[gm_params['reg_covar']]})
            results = results.append(entry,ignore_index=True)
            
        self.results_ = results        
        # Get the best cov type and its index within the dataframe
        best_idx = results['bic'].idxmin()

        self.bic_ = results.loc[best_idx,'bic']
        self.n_components_ = results.loc[best_idx,'n_components']
        self.covariance_type_ = results.loc[best_idx,'covariance_type']
        self.affinity_ = results.loc[best_idx,'affinity']
        self.linkage_ = results.loc[best_idx,'linkage']
        self.reg_covar_ = results.loc[best_idx,'reg_covar']
        self.ari_ = results.loc[best_idx,'ari']
        self.model_ = results.loc[best_idx,'model']

        return self