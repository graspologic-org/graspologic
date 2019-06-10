# GraSPy
[![arXiv shield](https://img.shields.io/badge/arXiv-1904.05329-red.svg?style=flat)](https://arxiv.org/abs/1904.05329)
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
- [Tutorials](#Tutorials)
- [License](#license)
- [Issues](https://github.com/neurodata/graspy/issues)

# Overview
A graph, or network, provides a mathematically intuitive representation of data with some sort of relationship between items. For example, a social network can be represented as a graph by considering all participants in the social network as nodes, with connections representing whether each pair of individuals in the network are friends with one another. Naively, one might apply traditional statistical techniques to a graph, which neglects the spatial arrangement of nodes within the network and is not utilizing all of the information present in the graph. In this package, we provide utilities and algorithms designed for the processing and analysis of graphs with specialized graph statistical algorithms.

# Documenation
The official documentation with usage is at https://graspy.neurodata.io/

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
This package is written for Python3. Currently, it is supported for Python 3.5, 3.6, and 3.7.

### Python Dependencies
`GraSPy` mainly depends on the Python scientific stack.
```
networkx
numpy
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

# Tutorials
Please visit the [tutorial section](https://graspy.neurodata.io/tutorial.html) in the official website

## License
This project is covered under the [Apache 2.0 License](https://github.com/neurodata/graspy/blob/master/LICENSE).
