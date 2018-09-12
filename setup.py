# adapted from https://github.com/neurodata/primitives-interfaces/blob/master/setup.py

import os
import sys
from setuptools import setup, find_packages
from setuptools.command.install import install
from subprocess import check_output, call
from sys import platform

PACKAGE_NAME = 'graphstats'
MINIMUM_PYTHON_VERSION = 3, 6
VERSION = '0.0.1'

def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


def read_package_variable(key):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, '__init__.py')
    with open(module_path) as module:
        for line in module:
            parts = line.strip().split(' ')
            if parts and parts[0] == key:
                return parts[-1].strip("'")
    assert False, "'{0}' not found in '{1}'".format(key, module_path)

check_python_version()
setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description='A function for graph statistics.',
    long_description=('A graph, or network, provides a mathematically intuitive representation '
      'of data with some sort of relationship between items. For example, a social network can be'
      'represented as a graph by considering all participants in the social network as nodes, with'
      ' connections representing whether each pair of individuals in the network are friends with one '
      'another. Naively, one might apply traditional statistical techniques to a graph, which neglects'
      ' the spatial arrangement of nodes within the network and is not utilizing all of the information'
      ' present in the graph. In this package, we provide utilities and algorithms designed for the'
      ' processing and analysis of graphs with specialized graph statistical algorithms.'),
    author='Eric Bridgeford, Jaewon Chung, Benjamin Pedigo, Bijan Varjavand, Brandon Duderstadt, Vivek Gopalakrishnan',
    author_email="ebridge2@jhu.edu",
    packages=find_packages(),
    install_requires=['numpy', 'networkx', 'sklearn'],
    url='https://github.com/neurodata/pygrapstats',
)
