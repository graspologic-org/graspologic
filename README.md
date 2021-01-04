# graspologic
[![Paper shield](https://img.shields.io/badge/JMLR-Paper-red)](http://www.jmlr.org/papers/volume20/19-490/19-490.pdf)
[![PyPI version](https://img.shields.io/pypi/v/graspologic.svg)](https://pypi.org/project/graspologic/)
[![Downloads shield](https://pepy.tech/badge/graspologic)](https://pepy.tech/project/graspologic)
[![Docs shield](https://img.shields.io/readthedocs/graspologic)](https://graspologic.readthedocs.io/)
![graspologic CI](https://github.com/microsoft/graspologic/workflows/graspologic%20CI/badge.svg)
[![codecov](https://codecov.io/gh/microsoft/graspologic/branch/dev/graph/badge.svg)](https://codecov.io/gh/microsoft/graspologic)
[![DOI](https://zenodo.org/badge/147768493.svg)](https://zenodo.org/badge/latestdoi/147768493)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


## `graspologic` is a package for graph statistical algorithms.

- [Overview](#overview)
- [Documentation](#documentation)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Contributing](#contributing)
- [License](#license)
- [Issues](#issues)

# Notice: `graspologic` is the merger project of `GraSPy` and `topologic`
We're actively merging these projects into one, but you may see some references to `graspy` or `topologic` from time to time in documentation and
issues. If you notice anything in the documentation referencing either graspy or topologic, please raise an issue (if one does not already exist)
noting the missed titles so we can address all of them.

# Overview
A graph, or network, provides a mathematically intuitive representation of data with some sort of relationship between items. For example, a social network can be represented as a graph by considering all participants in the social network as nodes, with connections representing whether each pair of individuals in the network are friends with one another. Naively, one might apply traditional statistical techniques to a graph, which neglects the spatial arrangement of nodes within the network and is not utilizing all of the information present in the graph. In this package, we provide utilities and algorithms designed for the processing and analysis of graphs with specialized graph statistical algorithms.

# Documentation
The official documentation with usage is at https://graspologic.readthedocs.io/en/latest/

Please visit the [tutorial section](https://graspologic.readthedocs.io/en/latest/tutorial.html) in the official website for more in depth usage.

# System Requirements
## Hardware requirements
`graspologic` package requires only a standard computer with enough RAM to support the in-memory operations.

## Software requirements
### OS Requirements
`graspologic` is tested on the following OSes:
- Linux x64
- macOS x64
- Windows 10 x64

And across the following versions of Python:
- 3.6 (x64)
- 3.7 (x64)
- 3.8 (x64)

If you try to use `graspologic` for a different platform than the ones listed and notice any unexpected behavior,
please feel free to [raise an issue](https://github.com/microsoft/graspologic/issues/new).  It's better for ourselves and our users
if we have concrete examples of things not working!

### Python Dependencies
`graspologic` has the following direct dependencies:
```
hyppo
matplotlib
networkx
numpy
POT
seaborn
scikit-learn
scipy
umap-learn
```

Developers of `graspologic` will also have the following dependencies:
```
black
ipykernel
ipython
myp
nbsphinx
numpydoc
pandoc
pytest
pytest-cov
sphinx
sphinxcontrib-rawfiles
sphinx-rtd-theme
```

Please note that `pandoc` will also [need to be installed for your system](https://pandoc.org/installing.html).

# Installation Guide
## Install from pip
```
pip install graspologic
```

## Install from Github
```
git clone https://github.com/microsoft/graspologic
cd graspologic
python3 -m venv venv
source venv/bin/activate
python3 setup.py install
```

# Contributing
We welcome contributions from anyone. Please see our [contribution guidelines](https://github.com/microsoft/graspologic/blob/dev/CONTRIBUTING.md) before making a pull request. Our
[issues](https://github.com/microsoft/graspologic/issues) page is full of places we could use help!
If you have an idea for an improvement not listed there, please
[make an issue](https://github.com/microsoft/graspologic/issues/new) first so you can discuss with the developers.

# License
This project is covered under the MIT License.

# Issues
We appreciate detailed bug reports and feature requests (though we appreciate pull requests even more!). Please visit our [issues](https://github.com/microsoft/graspologic/issues) page if you have questions or ideas.

# Citing `graspologic`
If you find `graspologic` useful in your work, please cite the package via the [GraSPy paper](http://www.jmlr.org/papers/volume20/19-490/19-490.pdf)

> Chung, J., Pedigo, B. D., Bridgeford, E. W., Varjavand, B. K., Helm, H. S., & Vogelstein, J. T. (2019). GraSPy: Graph Statistics in Python. Journal of Machine Learning Research, 20(158), 1-7.
