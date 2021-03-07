docs:
	sphinx-build -W -a docs/reference/ docs/reference/_build/html

tutorials:
	sphinx-build -W -a docs/tutorials/ docs/tutorials/_build/html

coverage:
	python -m pytest --co --cov=graspologic graspologic tests

lint:
	black --check --diff ./graspologic ./tests

test:
	pytest tests

type-check:
	mypy ./graspologic

validate: lint coverage docs test

.PHONY: docs tutorials coverage lint test type-check validate
