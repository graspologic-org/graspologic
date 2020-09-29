docs:
	sphinx-build -W -a docs/ docs/_build/html

coverage:
	python -m pytest --co --cov=graspologic

lint:
	black --check --diff ./graspologic ./tests

test:
	pytest tests

type-check:
	mypy ./graspologic

validate: lint coverage docs test

.PHONY: docs coverage lint test type-check validate