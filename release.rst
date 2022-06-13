[BUG] Fixes reproducibility in EdgeSwapper and adds to docs (#945)

* fix seeding and copying of outputs

* fix reproducibility testing logic

* add to documentation

* fix notebook

* windows can't handle the uint

* try to fix windows bug

* try to fix the fix

* try uint32 again
Added Degree Preserving Edge Swaps (#935)

* LPT tutorial render fix attempt

* lpt tutorial changes

* change order of inference functions return

* fix add_variance

* Revert "fix add_variance"

This reverts commit 47b0c4d2eb76e4e0a03ed02916f1711722bf2b4f.

* fix tutorials

* create function

* update edge swap

* remove initial loops condition

* less than two edges check

* typo

* add scipy function

* finish scipy function

* fix scipy function

* add scipy function

* remove print statements for sim

* add edge swap tests

* add scipy check_swaps method

* add swap functions to init

* fix functions

* refactor numpy and scipy into one function

* add functions to init

* make numba workable

* make edge swaps a class

* remove no_jit function

* fix tests

* fix pr errors

* add numba to mypy

* sort imports

* fix formatting

* make edge_list an instance variable

* sort imports

* add docstrings

* fix seed

* fix seed

* undirected implementation

* fix type of edge list

* format tests

* remove extra line

* add dpes tutorial

* fix errors

* fix numba import

* add dpes to tutorials index

* fix errors

* remove print statements from test

* Delete dpes.ipynb

* formatting of tests

* reformat tests

* fix tutorials

* fix seed type hinting

* black

* fix directed check

* add scipy test

* format

* format

* update language

* Update index.rst

* Update graspologic/models/edge_swaps.py

* Update docs

* update docs

* fix bug

* fix errors

* format:

* ensure build

* change type hinting

* remove some unused imports

* remove unused parameter

* simplify some input checking

* simplify checking logic more

* fix issue with not making a copy

* update to use import_graph

* manipulate LIL matrices

* remove unused imports

* simplify notebook and clear outputs

* try reworking to use JIT optionally as a function

* typo

* fix seed

* fix seed

* format

Co-authored-by: Benjamin Pedigo <benjamindpedigo@gmail.com>
fix mypy issue (#943)
Fixed loops bug in SBM and DCSBM model fitting  (#930)

* hotfix for #928

* updated formatting

* Update setup.cfg

* Update setup.cfg

Co-authored-by: Benjamin Pedigo <benjamindpedigo@gmail.com>
Error message in Leiden when given a multigraph was incorrect (#926)

The exception we raise when given a networkx graph if it's directed or a multigraph implies the only error condition is that it's a directed graph, when that isn't the case.
Updating the development status
Preparing for 1.0.1 or other versions
Fixed typos in models/er.py & models/sbm_estimators.py (#920)

Co-authored-by: Benjamin Pedigo <benjamindpedigo@gmail.com>
bump hyppo again
bump hyppo again
[WIP] Added release notes (#914)

* Update release.rst

* Update release.rst
Windows runners need to be explicitly told to use bash as the shell
