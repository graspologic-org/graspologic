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

- [ ] Followed the `[coding-guidelines](#guidelines)`.
- [ ] Given your pull request a helpful title that summarises what your contribution does. In some cases ``Fix <ISSUE TITLE>`` is enough. ``Fix #<ISSUE NUMBER>`` is not enough.
- [ ] All public methods should have informative docstrings with sample usage presented as doctests when appropriate.
- [ ] At least one paragraph of narrative documentation with links to references in the literature (with PDF links when possible) and the example.
- [ ] All functions and classes must have unit tests. These should include, at the very least, type checking and ensuring correct computation/outputs.
- [ ] Ensure all tests are passing locally using pytest. Install the necessary packages by:
```
$ pip install pytest pytest-cov
```
And then
```
$ pytest
```
or you can run pytest on a single test file by
```
$ pytest path/to/test.py
```
- [ ] Run an autoformatter. We use black and would like for you to format all files using ``black``. You can run the following lines to format your files.
```
$ pip install black
$ black path/to/module.py
```

#### Any other comments?

