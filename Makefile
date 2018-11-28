PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-ctags:
	rm -f tags

clean: clean-ctags
	rm -rf .pytest_cache
	find . -type f -name '*.pyc' | xargs rm -f
    find . -type d -name '__pycache__' | xargs rm -f

test-all:
    pytest -vsl

test: test-all

test-db:
    pytest -vsl ramp-database/

test-engine:
    pytest -vsl ramp-engine/

test-frontend:
    pytest -vsl databoard/

trailing-spaces:
	find databoard -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R databoard

code-analysis:
	flake8 databoard --ignore=E501,E211,E265 | grep -v __init__ | grep -v external
