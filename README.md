# GraSPy
[![Paper shield](https://img.shields.io/badge/JMLR-Paper-red)](http://www.jmlr.org/papers/volume20/19-490/19-490.pdf)
[![Downloads shield](https://img.shields.io/pypi/dm/graspy.svg)](https://pypi.org/project/graspy/)
[![Build Status](https://travis-ci.org/neurodata/graspy.svg?branch=master)](https://travis-ci.org/neurodata/graspy)
[![codecov](https://codecov.io/gh/neurodata/graspy/branch/master/graph/badge.svg)](https://codecov.io/gh/neurodata/graspy)
[![DOI](https://zenodo.org/badge/147768493.svg)](https://zenodo.org/badge/latestdoi/147768493)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


**Gra**ph **S**tatistics in **Py**thon is a package for graph statistical algorithms.

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Contributing](#contributing)
- [License](#license)
- [Issues](#issues)

# Overview
A graph, or network, provides a mathematically intuitive representation of data with some sort of relationship between items. For example, a social network can be represented as a graph by considering all participants in the social network as nodes, with connections representing whether each pair of individuals in the network are friends with one another. Naively, one might apply traditional statistical techniques to a graph, which neglects the spatial arrangement of nodes within the network and is not utilizing all of the information present in the graph. In this package, we provide utilities and algorithms designed for the processing and analysis of graphs with specialized graph statistical algorithms.

# Documentation
The official documentation with usage is at https://graspy.neurodata.io/

Please visit the [tutorial section](https://graspy.neurodata.io/tutorial.html) in the official website for more in depth usage.

# System Requirements
## Hardware requirements
`GraSPy` package requires only a standard computer with enough RAM to support the in-memory operations. 

## Software requirements
### OS Requirements
This package is supported for *Linux* and *macOS*. The package has been tested on the following systems:
+ Linux: Ubuntu 16.04
+ macOS: Mojave (10.14.1)
+ Windows: 10 

### Python Requirements
This package is written for Python3. Currently, it is supported for Python 3.6 and 3.7.

### Python Dependencies
`GraSPy` mainly depends on the Python scientific stack.
```
networkx
numpy
pandas
scikit-learn
scipy
seaborn
```

# Installation Guide
## Install from pip
```
pip install graspy
```

## Install from Github
```
git clone https://github.com/neurodata/graspy
cd graspy
python3 setup.py install
```

# Contributing
We welcome contributions from anyone. Please see our [contribution guidelines](https://graspy.neurodata.io/contributing.html) before making a pull request. Our 
[issues](https://github.com/neurodata/graspy/issues) page is full of places we could use help! 
If you have an idea for an improvement not listed there, please 
[make an issue](https://github.com/neurodata/graspy/issues/new) first so you can discuss with the 
developers. 

# License
This project is covered under the MIT License.

# Issues
We appreciate detailed bug reports and feature requests (though we appreciate pull requests even more!). Please visit our [issues](https://github.com/neurodata/graspy/issues) page if you have questions or ideas.

# Citing GraSPy
If you find GraSPy useful in your work, please cite the package via the [GraSPy paper](http://www.jmlr.org/papers/volume20/19-490/19-490.pdf)

> Chung, J., Pedigo, B. D., Bridgeford, E. W., Varjavand, B. K., Helm, H. S., & Vogelstein, J. T. (2019). GraSPy: Graph Statistics in Python. Journal of Machine Learning Research, 20(158), 1-7.
