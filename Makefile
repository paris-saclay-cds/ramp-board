all: clean inplace test

clean:
	rm -rf .pytest_cache
	find . -type f -name '*.pyc' | xargs rm -f
    find . -type d -name '__pycache__' | xargs rm -f

install:
    cd ramp-frontend && pip install . && cd ..
    cd ramp-database && pip install . && cd ..
    cd ramp-engine && pip install . && cd ..
    cd ramp-utils && pip install . && cd ..

inplace:
    cd ramp-frontend && pip install -e . && cd ..
    cd ramp-database && pip install -e . && cd ..
    cd ramp-engine && pip install -e . && cd ..
    cd ramp-utils && pip install -e . && cd ..

test-all:
    pytest -vsl

test: test-all

test-db:
    pytest -vsl ramp-database/

test-engine:
    pytest -vsl ramp-engine/

test-frontend:
    pytest -vsl ramp-frontend/

code-analysis:
	flake8 . --ignore=E501,E211,E265 | grep -v __init__ | grep -v external
