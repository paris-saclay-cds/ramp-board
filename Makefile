PYTHON ?= python
PYTEST ?= pytest

all: clean inplace test

clean:
	cd ramp-frontend && $(PYTHON) setup.py clean && cd ..
	cd ramp-database && $(PYTHON) setup.py clean && cd ..
	cd ramp-engine && $(PYTHON) setup.py clean && cd ..
	cd ramp-utils && $(PYTHON) setup.py clean && cd ..

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

upload-pypi:
	cd ramp-frontend && python setup.py sdist bdist_wheel && twine upload dist/* && cd ..
	cd ramp-database && python setup.py sdist bdist_wheel && twine upload dist/* && cd ..
	cd ramp-engine && python setup.py sdist bdist_wheel && twine upload dist/* && cd ..
	cd ramp-utils && python setup.py sdist bdist_wheel && twine upload dist/* && cd ..

clean-dist:
	rm -r ramp-frontend/dist
	rm -r ramp-database/dist
	rm -r ramp-engine/dist
	rm -r ramp-utils/dist
