PYTHON_FILES=*.py

# Specify the names of all executables to make.
PROG=init check precommit config flake8 black-fix black-check update install test
.PHONY: ${PROG}

default:
	@echo "An explicit target is required. Available options: ${PROG}"

init: install config

check: style black-check flake8

flake8:
	flake8 ${PYTHON_FILES}

style:
	isort --settings-path setup.cfg ${PYTHON_FILES}
	black --config pyproject.toml ${PYTHON_FILES}

black-check:
	isort --settings-path setup.cfg --check ${PYTHON_FILES}
	black --config pyproject.toml --check ${PYTHON_FILES}

install:
	pip install --upgrade pip setuptools wheel
	pip install --upgrade -r requirements.txt -r requirements-dev.txt

badges:
	-genbadge coverage -i coverage.xml -o images/coverage.svg
	-interrogate --generate-badge images/interrogate.svg --badge-format svg