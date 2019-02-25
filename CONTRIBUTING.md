# Contributing to GraSPy
(adopted from sklearn)

## How to Contribute
The preferred workflow for contributing to GraSPy is to fork the main repository on GitHub, clone, and develop on a branch. Steps:
1. Fork the [project repository](https://github.com/neurodata/GraSPy)
   by clicking on the 'Fork' button near the top right of the page. This creates
   a copy of the code under your GitHub user account. For more details on
   how to fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the GraSPy repo from your GitHub account to your local disk:

   ```bash
   $ git clone git@github.com:YourLogin/GraSPy.git
   $ cd GraSPy
   ```

3. Create a ``feature`` branch to hold your development changes:

   ```bash
   $ git checkout -b my-feature
   ```

   Always use a ``feature`` branch. It's good practice to never work on the ``master`` branch!

4. Develop the feature on your feature branch. Add changed files using ``git add`` and then ``git commit`` files:

   ```bash
   $ git add modified_files
   $ git commit
   ```

   to record your changes in Git, then push the changes to your GitHub account with:

   ```bash
   $ git push -u origin my-feature
   ```

## Pull Request Checklist
We recommended that your contribution complies with the following rules before you submit a pull request:
-  Follow the
   [coding-guidelines](#guidelines).
-  Give your pull request a helpful title that summarises what your
   contribution does. In some cases `Fix <ISSUE TITLE>` is enough.
   `Fix #<ISSUE NUMBER>` is not enough.
-  All public methods should have informative docstrings with sample
   usage presented as doctests when appropriate.
-  At least one paragraph of narrative documentation with links to
   references in the literature (with PDF links when possible) and
   the example.
-  Run an autoformatter. We use `black` and would like for you to format all files using `black`. You can run the following lines to format your files.
  ```bash
  $ pip install black
  $ black path/to/module.py
  ```

## Guidelines
### Coding Guidelines
Uniformly formatted code makes it easier to share code ownership. The pygraphstats package closely follows the official Python guidelines detailed in [PEP8](https://www.python.org/dev/peps/pep-0008/) that detail how code should be formatted and indented. Please read it and follow it.

### Docstring Guidelines
Properly formatted docstrings is required for documentation generation by Sphinx. The pygraphstats package closely follows the numpydoc guidelines. Please read and follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#overview) guidelines. Refer to the [example.py](https://numpydoc.readthedocs.io/en/latest/example.html#example) provided by numpydoc.
