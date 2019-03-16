PYTHON ?= python
PYTEST ?= pytest

all: clean inplace test

clean:
	$(PYTHON) setup.py clean

install:
	cd ramp-frontend && pip install . && cd ..
	cd ramp-database && pip install . && cd ..
	cd ramp-engine && pip install . && cd ..
	cd ramp-utils && pip install . && cd ..

in: inplace # just a shortcut
inplace:
	cd ramp-frontend && pip install -e . && cd ..
	cd ramp-database && pip install -e . && cd ..
	cd ramp-engine && pip install -e . && cd ..
	cd ramp-utils && pip install -e . && cd ..

test-all:
	$(PYTEST) -vsl .

test: test-all

test-db:
	$(PYTEST) -vsl ramp-database/

test-engine:
	$(PYTEST) -vsl ramp-engine/

test-frontend:
	$(PYTEST) -vsl ramp-frontend/

code-analysis:
	flake8 . --ignore=E501,E211,E265 | grep -v __init__ | grep -v external
