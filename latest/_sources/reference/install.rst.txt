Install
=======


Below we assume you have the default Python environment already configured on
your computer and you intend to install ``graspologic`` inside of it.  If you want
to create and work with Python virtual environments, please follow instructions
on `venv <https://docs.python.org/3/library/venv.html>`_ and `virtual
environments <http://docs.python-guide.org/en/latest/dev/virtualenvs/>`_.

First, make sure you have the latest version of ``pip`` (the Python package manager)
installed. If you do not, refer to the `Pip documentation
<https://pip.pypa.io/en/stable/installing/>`_ and install ``pip`` first.

Install the released version
----------------------------

Install the current release of ``graspologic`` with ``pip``::

    $ pip install graspologic

To upgrade to a newer release use the ``--upgrade`` flag::

    $ pip install --upgrade graspologic

If you do not have permission to install software systemwide, you can
install into your user directory using the ``--user`` flag::

    $ pip install --user graspologic

Alternatively, you can manually download ``graspologic`` from
`PyPI <https://pypi.python.org/pypi/graspologic>`_.
To install in this way, navigate to the download folder in your terminal and run
`pip install graspologic-VERSION.tar.gz`

Python package dependencies
---------------------------
graspologic requires the following packages:

- networkx
- numpy
- pandas
- scikit-learn
- scipy
- seaborn


Hardware requirements
---------------------
`graspologic` package requires only a standard computer with enough RAM to support the in-memory operations. 

OS Requirements
---------------
This package is supported for *Linux* and *macOS*. However, the package has been tested on the following systems:

- Linux: N/A
- macOS: N/A
- Windows: N/A


Testing
-------
graspologic uses the Python ``pytest`` testing package.  If you don't already have
that package installed, follow the directions on the `pytest homepage
<https://docs.pytest.org/en/latest/>`_.
