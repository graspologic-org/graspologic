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
contributors (see our [Roles page](https://github.com/graspologic-org/graspologic/blob/dev/ROLES.md)).

In case you experience issues using this package, do not hesitate to submit a ticket to our 
[Issue Tracker](https://github.com/graspologic-org/graspologic/issues).  You are also welcome to post feature requests or pull requests.

It is recommended to check that your issue complies with the following rules before submitting:

- Verify that your issue is not being currently addressed by other 
  [issues](https://github.com/graspologic-org/graspologic/issues?q=) or 
  [pull requests](https://github.com/graspologic-org/graspologic/pulls?q=).

- If you are submitting a bug report, we strongly encourage you to follow the guidelines in 
  [How to create an actionable bug report](#how-to-create-an-actionable-bug-report)

## How to create an actionable bug report

When you submit an issue to [Github](https://github.com/graspologic-org/graspologic/issues), please do your best to
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

# Contributing code

## Prerequisites
* [Python 3.9+](https://www.python.org/)
* [Poetry 1.8.2+](https://python-poetry.org/docs/)

## Setting up for development

The preferred workflow for contributing to `graspologic` is to fork the main repository on GitHub, clone, and develop on a
branch using a virtual environment. Steps: 

1. Fork the [project repository](https://github.com/graspologic-org/graspologic) by clicking on the ‘Fork’ button near the top
   right of the page. This creates a copy of the code under your GitHub user account. For more details on how to
   fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the `graspologic` repo from your GitHub account to your local disk. Do this by typing the following into command prompt or the equivelant on your operating system:

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

4. From the project root, run `poetry install`, which will create a virtual environment and install necessary dependencies.

## Code Changes

### Writing Code
- Make sure to follow the coding guidelines outlined below:
  - Uniformly formatted code makes it easier to share code ownership. Graspologic package closely follows the official Python guidelines detailed in [PEP8](https://www.python.org/dev/peps/pep-0008/) that detail how code should be formatted and indented. Please read it and follow it.
  - In order to make sure all code is formatted seamlessly and uniformly, we use [black](https://github.com/psf/black) to automatically format our code.
  - All new functions should have PEP-compliant type hints and [@beartype](https://github.com/beartype/beartype) decorator.  This allows us a reasonable level of confidence that arguments passed into the API are what we expect them to be without sacrificing runtime speed.
- All public methods should have informative [`docstrings`](https://github.com/graspologic-org/graspologic/blob/dev/CONTRIBUTING.md#docstring-guidelines) with sample usage presented as doctests when appropriate.
  - Properly formatted docstrings are required for documentation generation by [sphinx](https://www.sphinx-doc.org/en/master/usage/index.html). The graspologic package closely 
follows the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#overview) guidelines. Please read and follow the 
numpydoc guidelines. Refer to the 
[example.py](https://numpydoc.readthedocs.io/en/latest/example.html#example) provided by numpydoc.
- If proposing a new method, include at least one paragraph of narrative documentation with links to references in the literature (with PDF links when possible) and the example.
- If your feature is complex enough, consider creating a Jupyter notebook tutorial to illustrate its use instead. Tutorial Jupyter notebooks can be added to the docs [here](https://github.com/graspologic-org/graspologic/tree/dev/docs/tutorials).
- All functions and classes should be rigorously typed with Python 3.5+ [`typehinting`](https://docs.python.org/3/library/typing.html). 
- All functions and classes must have unit tests. These should include, at the very least, type checking and ensuring correct computation/outputs.

  - It's important to write unit tests for your bug fix and your features. When fixing a bug, first create a test that explicitly exercises the bug and results in a test case failure.  Then create the fix and run the test again to verify your results.

  - For new features, we advocate using [TDD](https://en.wikipedia.org/wiki/Test-driven_development) wherever possible.

### Checking code

After you have made changes to the `graspologic` code, you should use several
tools to help make sure your changes meet the standards for our repository.

#### Code formatting
Please use `black` and `isort` so that the format of your code is compatible with our project. Format your code prior to committing using one of the following methods:
```bash
poetry run poe format
```

#### Type checking
Validate your typehinting by running:
```bash
poetry run poe type_check
```

#### Unit testing 
To check if your code runs correctly, we recommend using unit testing that locally tests your code by implementing test cases. Execute these unit tests by running:
```bash
poetry run poe tests
```

#### Creating documentation
Build the documentation with the use of [sphinx](https://www.sphinx-doc.org/en/master/usage/index.html) by running:
```bash
poetry run poe docs
```
Please verify that the built documentation looks appropriate. You can view the `html`
from the `docs/_build/html` folder; click on `index.html` to see what the homepage would
look like and navigate from there.

If you have made any changes that could affect the tutorials, please also build them.
This can take a bit longer because the code in each notebook actually needs to execute.
You can build the documentation and tutorials by running:
```bash
poetry run poe docsWithTutorials
```

## Publishing Changes

### Useful Git Commands
When working on a new feature, develop the feature on your feature branch. Add changed files using `git add` and then `git commit` files:

   ```bash
   git add modified_files
   git commit -m "your commit message"
   ```

   After making all local changes, you will want to push your changes to your fork:
   ```bash
   git push -u origin my-feature
   ```

### Creating a pull request

We recommend that your pull request complies with the following rules before it is submitted: 

- Make sure that the base repository and head repository, as well as the "base" file and "compare" file, are pointing to the correct locations
- Give your pull request (PR) a helpful title, set in the past tense, that summarizes what your contribution does. We are using PR titles to automatically generate release notes; examples of helpful PR title formats include: 
   - `Added Feature[Set] {Title|Short Descriptor} in ModuleOrPackageName`
   - `Fixed bug in [ClassName.method_name|ModuleOrPackageName.function_name] where ShortDescription`
   - `Updated [ClassName[.method_name]|ModuleOrPackageName.function_name] to ShortDescription`
- Link your pull request to the issue (see: [closing keywords](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue) for an easy way of linking your issue)
- Include a brief description of the changes you made in the code in the "write" box provided in the pull request page

Once submitted, your PR will undergo automated tests that ensure its compilability and compatibility with our project. For debugging tests that raise errors online but passed locally, one can look at [this file](https://github.com/graspologic-org/graspologic/blob/dev/.github/workflows/build.yml) to see Github's exact execution.



