docs:
	sphinx-build -W -a docs/reference/ docs/reference/_build/html

tutorials:
	sphinx-build -W -a docs/tutorials/ docs/tutorials/_build/html

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

.PHONY: docs tutorials coverage lint test type-check validate
