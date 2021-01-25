# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build/html

.PHONY: help clean html


# Put it first so that "make" without argument is like "make help".
help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html      to make standalone HTML files"

clean:
	-rm -rf _build/*

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) "$(SOURCEDIR)" "$(BUILDDIR)"
	@echo
	@echo "Build finished. The HTML pages are in build/html."

