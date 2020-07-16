---
title: 'AutoGMM: Automatic Gaussian Mixture Modeling in Python'
tags:
  - Python
  - clustering
  - gaussian mixture models
  - mclust
  - unsupervised
authors:
  - name: Thomas L. Athey
    orcid: 0000-0002-8987-3486
    affiliation: 1
  - name: Joshua T. Vogelstein^[Corresponding Author]
    affiliation: 1
affiliations:
 - name: Department of Biomedical Engineering Johns Hopkins University
   index: 1
date: 13 July 2020
bibliography: refs_joss.bib

---

# Summary

*AutoGMM* is a Python algorithm for automatic Gaussian mixture modeling. It builds upon *scikit-learn*'s AgglomerativeClustering and GaussianMixture classes [@sklearn], with certain modifications to make the results more stable. 

\autoref{fig:example} shows an example application of *AutoGMM*, on the Wisconsin Breast Cancer dataset from the UCI Machine Learning Repository [@bc]. It is compared to *mclust* [@mclust] and the GaussianCluster class in *GraSPy* [@graspy]. *AutoGMM* is available as a class in the cluster module of *GraSPy*.

# Statement of need 

Gaussian mixture modeling is a fundamental tool in clustering, as well as discriminant analysis and semiparametric density estimation. However, estimating the optimal model for any given number of components is an NP-hard problem, and estimating the number of components is in some respects an even harder problem. 
In R, a popular package called *mclust* addresses both of these problems [@mclust].  

However,  Python has lacked such a package. We therefore introduce *AutoGMM*, which is freely available and therefore further shrinks the gap between functionality of R and Python for data science.

# Mathematics

*AutoGMM* performs an initial clustering (using k-means, agglomerative clustering, or random initialization), then fits a Gaussian mixture model using Expectation-Maximization. The algorithm sweeps through combinations of clustering options such as Gaussian covariance constraints and number of clusters. Each combination is evaluated with the Bayesian Information Criterion, defined as $2ln(\hat{L}) - p \cdot \ln(n)$ where $\hat{L}$ is the maximized data likelihood, $p$ is the number of parameters, and $n$ is the number of data points [@bic].

The data likelihood in Expectation-Maximization of Gaussian mixture models can diverge if one of the Gaussians becomes concentrated around a single data point. When this occurs, *AutoGMM* reruns the clustering by adding a regularization factor to the diagonal of the covariance matrices. This ensures that the estimated covariances have positive eigenvalues, without affecting the eigenvectors. 


# Figures

![Three different clustering algorithms were applied to the Breast Cancer Wisconsin Data Set from the UCI Machine Learning Repository [@bc]. The true clustering is shown in panel a, the *AutoGMM* result in panel b, *mclust* in panel c [@mclust], and the GaussianCluster class in *GraSPy* (called GraSPyclust here) in panel d [@graspy]. All algorithms were run with default parameters. There were two true underlying clusters. AutoGMM and mclust both predicted three underlying clusters, and produced similar clusterings. GraSPyclust predicted four underlying clusters. \label{fig:example}](example.png)
	
# References
