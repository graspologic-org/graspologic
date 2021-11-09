# -*- coding: utf-8 -*-
#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys

sys.path.append(os.path.abspath('./sphinx-ext/'))
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

project = "graspologic"
copyright = "2020"
authors = "Microsoft, NeuroData"

realpath = os.path.realpath(__file__)
dir_realpath = os.path.dirname(realpath)
sys.path.append(dir_realpath)

import graspologic

version = graspologic.__version__
# Append "dev" and the github run to the version when on the dev branch
if os.environ.get("GITHUB_REF", "") == "refs/heads/dev":
    version = f"{version}dev{os.environ['GITHUB_RUN_ID']}"

release = version

# -- Extension configuration -------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "numpydoc",
    "sphinx.ext.ifconfig",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "toctree_filter",
]

# -- numpydoc
# Below is needed to prevent errors
numpydoc_show_class_members = False
numpydoc_attributes_as_param_list = True
numpydoc_use_blockquotes = True

# -- sphinx.ext.autosummary
autosummary_generate = True

# -- sphinx.ext.autodoc
autoclass_content = "both"
autodoc_default_flags = ["members", "inherited-members"]
autodoc_member_order = "bysource"  # default is alphabetical

# -- sphinx.ext.intersphinx
intersphinx_mapping = {
    "anytree": ("https://anytree.readthedocs.io/en/latest/", None),
    "hyppo": ("https://hyppo.neurodata.io", None),
    "joblib": ("https://joblib.readthedocs.io/en/latest/", None),
    "matplotlib": ("https://matplotlib.org", None),
    "networkx": ("https://networkx.org/documentation/stable", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference", None),
    "seaborn": ("https://seaborn.pydata.org", None),
    "sklearn": ("https://scikit-learn.org/dev", None),
}

# -- sphinx options ----------------------------------------------------------
source_suffix = ".rst"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "tutorials"]
toc_filter_exclude = ['tutorials/index']
master_doc = "index"
source_encoding = "utf-8"
if tags.has("build_tutorials"):
    # Tutorials are excluded by default.  Remove the exclusion since we want to build the tutorials
    exclude_patterns.remove("tutorials")
    toc_filter_exclude = []

# -- Options for HTML output -------------------------------------------------
# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]
html_static_path = []
modindex_common_prefix = ["graspologic."]

pygments_style = "sphinx"
smartquotes = False

# Use RTD Theme
import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
html_theme_options = {
    #'includehidden': False,
    "navigation_depth": 2,
    "collapse_navigation": False,
    "navigation_depth": 3,
}

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "graspologicdoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "graspologic.tex", "graspologic Documentation", authors, "manual")
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "graspologic", "graspologic Documentation", [authors], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "graspologic",
        "graspologic Documentation",
        authors,
        "graspologic",
        "One line description of project.",
        "Miscellaneous",
    )
]

# -- Options for Epub output -------------------------------------------------

# Bibliographic Dublin Core info.
epub_title = project

# The unique identifier of the text. This can be a ISBN number
# or the project homepage.
#
# epub_identifier = ''

# A unique identification for the text.
#
# epub_uid = ''

# A list of files that should not be packed into the epub file.
epub_exclude_files = ["search.html"]
