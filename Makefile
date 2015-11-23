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

test:
	nosetests databoard/tests
	# fab publish_test:iris_local
	# cd /tmp/databoard_local/databoard_iris_8080; fab test_ramp

	# fab publish_test:boston_housing_local
	# cd /tmp/databoard_local/databoard_boston_housing_8080; fab test_ramp

test-all: test
	fab publish_test:amadeus_local
	cd /tmp/databoard_local/databoard_amadeus_8080; fab test_ramp

	fab publish_test:el_nino_bagged_cv_future_local
	cd /tmp/databoard_local/databoard_el_nino_bagged_cv_future_8080; fab test_ramp

	fab publish_test:el_nino_block_cv_local
	cd /tmp/databoard_local/databoard_el_nino_block_cv_8080; fab test_ramp

	fab publish_test:kaggle_otto_local
	cd /tmp/databoard_local/databoard_kaggle_otto_8080; fab test_ramp

	fab publish_test:mortality_prediction_local
	cd /tmp/databoard_local/databoard_mortality_prediction_8080; fab test_ramp

	fab publish_test:pollenating_insects_local
	cd /tmp/databoard_local/databoard_pollenating_insects_8080; fab test_ramp

	fab publish_test:variable_stars_local
	cd /tmp/databoard_local/databoard_variable_stars_8080; fab test_ramp

trailing-spaces:
	find databoard -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R databoard

code-analysis:
	flake8 databoard --ignore=E501,E211,E265 | grep -v __init__ | grep -v external
