clean:
	rm -rf docs/_build

docs:
	sphinx-build -W -a docs/ docs/_build/html

docsWithTutorials:
	sphinx-build -W -t build_tutorials -a docs/ docs/_build/html

coverage:
	python -m pytest --co --cov=graspologic graspologic tests

lint:
	black --check --diff ./graspologic ./tests
	isort --check-only ./graspologic ./tests
format:
	black ./graspologic ./tests
	isort ./graspologic ./tests
test:
	pytest tests

fast-test:
	pytest tests --ignore=tests/test_latentdistributiontest.py --ignore=tests/test_latentpositiontest.py

type-check:
	mypy ./graspologic

validate: lint coverage docs test

.PHONY: docs docsWithTutorials coverage lint format test type-check validate
