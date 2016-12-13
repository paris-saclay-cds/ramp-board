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
	nosetests databoard/tests
	fab publish_local_test
	cd /tmp/datacamp/databoard; fab test_setup; fab test_setup

test-all: test
	fab publish_test:amadeus_local
	cd /tmp/databoard_amadeus_8080; fab test_ramp

	fab publish_test:el_nino_bagged_cv_future_local
	cd /tmp/databoard_el_nino_bagged_cv_future_8080; fab test_ramp

	fab publish_test:el_nino_block_cv_local
	cd /tmp/databoard_el_nino_block_cv_8080; fab test_ramp

	fab publish_test:kaggle_otto_local
	cd /tmp/databoard_kaggle_otto_8080; fab test_ramp

	fab publish_test:mortality_prediction_local
	cd /tmp/databoard_mortality_prediction_8080; fab test_ramp

	fab publish_test:pollenating_insects_local
	cd /tmp/databoard_pollenating_insects_8080; fab test_ramp

	fab publish_test:variable_stars_local
	cd /tmp/databoard_variable_stars_8080; fab test_ramp

trailing-spaces:
	find databoard -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R databoard

code-analysis:
	flake8 databoard --ignore=E501,E211,E265 | grep -v __init__ | grep -v external
