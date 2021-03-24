.PHONY: docs


# Put it first so that "make" without argument is like "make help".
docs:
	cd reference && pip install -r requirements.txt && make html
	cd tutorials && pip install -r requirements.txt && make html
	mkdir -p _build/html
	cp -r reference/_build/html _build/html/reference
	cp -r tutorials/_build/html _build/html/tutorials

