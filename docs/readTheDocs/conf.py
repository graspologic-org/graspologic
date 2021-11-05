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


# -- Project information -----------------------------------------------------

project = "graspologic"
copyright = "2020"
authors = "Microsoft, NeuroData"

version = "latest"
release = "latest"

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


# -- sphinx options ----------------------------------------------------------
source_suffix = ".rst"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "tutorials"]
toc_filter_exclude = ['tutorials/index']
master_doc = "index"
source_encoding = "utf-8"

# -- Options for HTML output -------------------------------------------------
# Add any paths that contain templates here, relative to this directory.
templates_path = ["../_templates"]
html_static_path = []

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


# -- Options for Texinfo output ----------------------------------------------


# Bibliographic Dublin Core info.
epub_title = project
