# Copyright (c) Microsoft Corporation and contributors.
# Licensed under the MIT License.

import os
import sys
from setuptools import setup, find_packages
from typing import Tuple


def package_metadata() -> Tuple[str, str]:
    sys.path.insert(0, os.path.join("graspy", "version"))  # TODO: #454 Change path in https://github.com/microsoft/graspologic/issues/454
    from version import name, version
    sys.path.pop(0)

    version_path = os.path.join("graspy", "version", "version.txt")
    with open(version_path, "w") as version_file:
        _b = version_file.write(f"{version}")
    return name, version


PACKAGE_NAME, VERSION = package_metadata()

DESCRIPTION = "A set of python modules for graph statistics"
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = ("Eric Bridgeford, Jaewon Chung, Benjamin Pedigo, Bijan Varjavand",)
AUTHOR_EMAIL = "j1c@jhu.edu"
URL = "https://github.com/neurodata/graspy"
MINIMUM_PYTHON_VERSION = 3, 6  # Minimum of Python 3.6
REQUIRED_PACKAGES = [
    "networkx>=2.1",
    "numpy>=1.8.1",
    "scikit-learn>=0.19.1",
    "scipy>=1.4.0",
    "seaborn>=0.9.0",
    "matplotlib>=3.0.0",
    "hyppo>=0.1.3",
]


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION:
        sys.exit("Python {}.{}+ is required.".format(*MINIMUM_PYTHON_VERSION))


check_python_version()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    maintainer="Dwayne Pryce",
    maintainer_email="dwpryce@microsoft.com",
    install_requires=REQUIRED_PACKAGES,
    url=URL,
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(exclude=["tests", "tests.*", "tests/*"]),
    include_package_data=True,
    package_data={'version': [os.path.join('graspy', 'version', 'version.txt')]},  # TODO: #454 Also needs changed by https://github.com/microsoft/graspologic/issues/454,
)
