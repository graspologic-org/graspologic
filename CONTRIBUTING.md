# Contributing

This project welcomes contributions and suggestions. Most contributions require you to
agree to a Contributor License Agreement (CLA) declaring that you have the right to,
and actually do, grant us the rights to use your contribution. For details, visit
https://cla.microsoft.com.

When you submit a pull request, a CLA-bot will automatically determine whether you need
to provide a CLA and decorate the PR appropriately (e.g., label, comment). Simply follow the
instructions provided by the bot. You will only need to do this once across all repositories using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/)
or contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

# Issue Submission (Bug or Feature)

We use GitHub issues to track all bugs and feature requests; feel free to open an issue if you have found a bug or wish 
to see a feature implemented. Please also feel free to tag one of the core 
contributors (see our [Roles page](https://github.com/microsoft/graspologic/blob/dev/ROLES.md)).

In case you experience issues using this package, do not hesitate to submit a ticket to our 
[Issue Tracker](https://github.com/microsoft/graspologic/issues).  You are also welcome to post feature requests or pull
requests.

It is recommended to check that your issue complies with the following rules before submitting:

- Verify that your issue is not being currently addressed by other 
  [issues](https://github.com/microsoft/graspologic/issues?q=) or 
  [pull requests](https://github.com/microsoft/graspologic/pulls?q=).

- If you are submitting a bug report, we strongly encourage you to follow the guidelines in 
  [How to create an actionable bug report](#how-to-create-an-actionable-bug-report)

## How to create an actionable bug report

When you submit an issue to [Github](https://github.com/microsoft/graspologic/issues), please do your best to
follow these guidelines! This will make it a lot faster for us to respond to your issue.

- The ideal bug report contains a **short reproducible code snippet**, this way
  anyone can try to reproduce the bug easily (see [this](https://stackoverflow.com/help/mcve) for more details). 
  If your snippet is longer than around 50 lines, please link to a [gist](https://gist.github.com) or a github repo.

- If not feasible to include a reproducible snippet, please be specific about
  what **estimators and/or functions are involved and the shape of the data**.

- If an exception is raised, please **provide the full traceback**.

- Please include your **operating system type and version number**, as well as
  your **Python and graspologic versions**. This information
  can be found by running the following code snippet:

    ```python
    import platform; print(platform.platform())
    import sys; print(f"Python {sys.version}")
    import graspologic; print(f"graspologic {graspologic.__version__}")
    ```

- Please ensure all **code snippets and error messages are formatted in
  appropriate code blocks**.  See 
  [Creating and highlighting code blocks](https://help.github.com/articles/creating-and-highlighting-code-blocks)
  for more details.

# Contributing Code

## Git workflow

The preferred workflow for contributing to Graspologic is to fork the main repository on GitHub, clone, and develop on a
branch. Steps: 

1. Fork the [project repository](https://github.com/microsoft/graspologic) by clicking on the ‘Fork’ button near the top
   right of the page. This creates a copy of the code under your GitHub user account. For more details on how to
   fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the Graspologic repo from your GitHub account to your local disk:

   ```bash
   git clone git@github.com:YourGithubAccount/graspologic.git
   cd graspologic
   ```

3. Create a feature branch to hold your development changes:

   ```bash
   git checkout -b my-feature
   ```

   Always use a `feature` branch. Pull requests directly to either `dev` or `main` will be rejected
   until you create a feature branch based on `dev`.

4. Unit testing

   It's important to write unit tests for your bug fix and your features. When fixing a bug, first create a test that explicitly exercises the bug and results in a test case failure.  Then create the fix and run the test again to verify your results.

   For new features, we advocate using [TDD](https://en.wikipedia.org/wiki/Test-driven_development) wherever possible.

   We also explicitly ask that you hew toward the `unittest` Python module for conformance.  This will ensure it plays nicely with most common IDEs on the market.

5. Code formatting:
   It's important to us that you follow the standards of our project.  Please use `black` and `isort` prior to
   committing.

   ```bash
   # Run "black" and "isort" using Make
   make format
   ```
   OR
   ```bash
   black graspologic/ tests/
   isort graspologic/ tests/
   ```

6. Develop the feature on your feature branch. Add changed files using `git add` and then `git commit` files:

   ```bash
   git add modified_files
   git commit
   ```

   After making all local changes, you will want to push your changes to your fork:
   ```bash
   git push -u origin my-feature
   ```

## Local Developer Setup
1. Make sure you have a compatible version of Python 3 installed
2. From the project root, create a virtual environment and install all development dependencies.  This example uses Python 3.8 but you may use any Python version supported by graspologic.

   **Bash**
    ```bash
   # Create virtual environment
   python3.8 -m venv venv
   
   # Activate the virtual environment
   source venv/bin/activate
   
   # Install development dependencies
   pip install -U pip setuptools
   pip install -r requirements.txt
   ```
   **CMD (Windows)**
    ```cmd
   rem Create virtual environment.  Depending on your installation you might need "py -3.8 -m venv venv" instead
   python3.8 -m venv venv
   
   rem Activate the virtual environment
   .\venv\Scripts\activate.bat
   
   rem Install development dependencies
   pip install -U pip setuptools
   pip install -r requirements.txt
   ```
   **PowerShell**
    ```powershell
   # Create virtual environment
   python3.8 -m venv venv
   
   # Activate the virtual environment
   .\venv\Scripts\Activate.ps1
   
   # Install development dependencies
   pip install -U pip setuptools
   pip install -r requirements.txt
   ```
3. Start playing with Graspologic code!

## Pull Request Checklist

We recommended that your contribution complies with the following rules before you submit a pull request: 

- Follow the [coding-guidelines](#guidelines). 
- Give your pull request (PR) a helpful title that summarizes what your contribution does. We are using PR titles to automatically generate release notes; examples of helpful PR title formats include: 
   - `Added Feature[Set] {Title|Short Descriptor} in ModuleOrPackageName`
   - `Fixed bug in [ClassName.method_name|ModuleOrPackageName.function_name] where ShortDescription`
   - `Updated [ClassName[.method_name]|ModuleOrPackageName.function_name] to ShortDescription`
- Link your pull request to the issue (see: 
  [closing keywords](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue) 
  for an easy way of linking your issue)
- All public methods should have informative docstrings with sample usage presented as doctests when appropriate. 
- At least one paragraph of narrative documentation with links to references in the literature (with PDF links when 
  possible) and the example. 
- If your feature is complex enough that a doctest is insufficient to fully showcase the utility, consider creating a 
  Jupyter notebook to illustrate use instead
- All functions and classes must have unit tests. These should include, at the very least, type checking and ensuring 
  correct computation/outputs.
- All functions and classes should be rigorously typed with Python 3.5+ 
  [`typehinting`](https://docs.python.org/3/library/typing.html). Validate your typehinting by running `mypy ./graspologic`
- All code should be automatically formatted by `black`. You can run this formatter by calling:
  ```bash
  pip install black isort
  black path/to/your_module.py
  isort path/to/your_module.py
  ```
- Ensure all tests are passing locally using `pytest`. Install the necessary
  packages by: 

  ```bash
  pip install pytest pytest-cov
  pytest
  ```

# Guidelines

## Coding Guidelines

Uniformly formatted code makes it easier to share code ownership. Graspologic package closely follows the official 
Python guidelines detailed in [PEP8](https://www.python.org/dev/peps/pep-0008/) that detail how code should be 
formatted and indented. Please read it and follow it.

All new functions should have PEP-compliant type hints and "@beartype" annotations.  This allows us a reasonable level 
of confidence that arguments passed into the API are what we expect them to be without sacrificing runtime speed.  See 
https://github.com/beartype/beartype for more information.

## Docstring Guidelines

Properly formatted docstrings are required for documentation generation by Sphinx. The graspologic package closely 
follows the numpydoc guidelines. Please read and follow the 
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#overview) guidelines. Refer to the 
[example.py](https://numpydoc.readthedocs.io/en/latest/example.html#example) provided by numpydoc.
