# GraSPy
[![Build Status](https://travis-ci.org/neurodata/graspy.svg?branch=master)](https://travis-ci.org/neurodata/graspy)
[![codecov](https://codecov.io/gh/neurodata/graspy/branch/master/graph/badge.svg)](https://codecov.io/gh/neurodata/graspy)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)


**Gra**ph **S**tatistics in **Py**thon is a package for graph statistical algorithms.

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [License](#license)
- [Issues](https://github.com/neurodata/graspy/issues)

# Overview
A graph, or network, provides a mathematically intuitive representation of data with some sort of relationship between items. For example, a social network can be represented as a graph by considering all participants in the social network as nodes, with connections representing whether each pair of individuals in the network are friends with one another. Naively, one might apply traditional statistical techniques to a graph, which neglects the spatial arrangement of nodes within the network and is not utilizing all of the information present in the graph. In this package, we provide utilities and algorithms designed for the processing and analysis of graphs with specialized graph statistical algorithms.

# System Requirements
## Hardware requirements
`GraSPy` package requires only a standard computer with enough RAM to support the in-memory operations. 

## Software requirements
### OS Requirements
This package is supported for *Linux* and *macOS*. The package has been tested on the following systems:
+ Linux: N/A
+ macOS: N/A
+ Windows: N/A

### Python Requirements
This package is written for Python3. Currently, it is supported for Python 3.5, 3.6, and 3.7.

### Python Dependencies
```
networkx
numpy
scikit-learn
scipy
seaborn
```

# Installation Guide
## Install from pip
Coming soon!

## Install from Github
```
git clone https://github.com/neurodata/graspy
cd graspy
python3 setup.py install
```

# Demo
List Jupyter notebooks here.

## License
This project is covered under the [Apache 2.0 License](https://github.com/neurodata/graspy/blob/master/LICENSE).
