PYTHON ?= python
NOSETESTS ?= nosetests
CTAGS ?= ctags

all: clean inplace test

clean-ctags:
	rm -f tags

clean: clean-ctags
	# $(PYTHON) setup.py clean
	rm -rf dist
	find . -name "*.pyc" | xargs rm -f

in: inplace # just a shortcut
inplace:
	# to avoid errors in 0.15 upgrade
	$(PYTHON) setup.py build_ext -i

test-local:  # test using a local sqlite database
	export DATABOARD_DB_URL_TEST=sqlite:////tmp/databoard_test.db; \
	export DATABOARD_DB_URL=sqlite:////tmp/databoard_test.db; \
	export DATABOARD_TEST=True; \
	make test

test:
	# nosetests databoard/tests
	fab deploy_locally
	fab test_problem:iris,kegl
	fab test_problem:boston_housing,kegl
	fab test_keywords
	fab test_make_event_admin

test-all: test
	# nosetests databoard/tests
	fab deploy_locally
	fab test_problem:iris,kegl
	fab test_problem:boston_housing,kegl
	fab test_keywords
	fab test_make_event_admin
	fab test_problem:titanic,kegl
	fab test_problem:epidemium2_cancer_mortality,kegl
	fab test_problem:HEP_detector_anomalies,kegl
	fab test_problem:drug_spectra,kegl

trailing-spaces:
	find databoard -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R databoard

code-analysis:
	flake8 databoard --ignore=E501,E211,E265 | grep -v __init__ | grep -v external
