<!--
Thanks for contributing a pull request! Please ensure you have taken a look at
the contribution guidelines: https://github.com/neurodata/graspy/blob/master/CONTRIBUTING.md#pull-request-checklist
-->

#### Reference Issues/PRs
<!--
Example: Fixes #1234. See also #3456.
Please use keywords (e.g., Fixes) to create link to the issues or pull requests
you resolved, so that they will automatically be closed when your pull request
is merged. See https://github.com/blog/1506-closing-issues-via-pull-requests
-->


#### What does this implement/fix? Explain your changes.


#### Checklist

- [ ] Followed the [coding-guidelines](#guidelines).
- [ ] Pull request has a helpful title that summarizes what your contribution does.
- [ ] Linked your pull request to the issue (see: [closing keywords](https://docs.github.com/en/free-pro-team@latest/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue) for an easy way of linking your issue)
- [ ] All public methods have informative docstrings with sample usage presented as doctests when appropriate.
- [ ] It has at least one paragraph of narrative documentation with links to references in the literature (with PDF links when possible) and the example.
- [ ] If your feature is complex enough that a doctest is insufficient to fully showcase the utility, consider creating a Jupyter notebook to illustrate use instead.
- [ ] All functions and classes have unit tests. Required tests include type checking, bounds validation, correct computation/outputs, and all conceivable edge cases.
- [ ] All functions and classes has been rigorously typed with Python 3.6+ 
  [`typehinting`](https://docs.python.org/3/library/typing.html). Validate your typehinting by running `mypy ./graspologic`
- [ ] All code should be automatically formatted by `black`. You can run this formatter by calling:
  ```bash
  pip install black
  black path/to/your_module.py
  ```
- [ ] Ensure all tests are passing locally using `pytest`. Install the necessary
  packages by: 

  ```bash
  pip install pytest pytest-cov
  pytest
  ```

#### Any other comments?

