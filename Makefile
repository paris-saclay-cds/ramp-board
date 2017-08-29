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
	fab send_password_mail:kegl,bla
	echo 'name,password\nkegl,bla' > pwds.csv
	fab send_password_mails:pwds.csv
	rm -rf pwds.csv


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
	fab test_problem:air_passengers,kegl
	fab test_problem:HEP_tracking,kegl
	fab test_problem:MNIST,kegl

test-heavy:
	fab deploy_locally
	fab test_problem:el_nino,kegl
	fab test_problem:sea_ice,kegl

trailing-spaces:
	find databoard -name "*.py" -exec perl -pi -e 's/[ \t]*$$//' {} \;

ctags:
	# make tags for symbol based navigation in emacs and vim
	# Install with: sudo apt-get install exuberant-ctags
	$(CTAGS) --python-kinds=-i -R databoard

code-analysis:
	flake8 databoard --ignore=E501,E211,E265 | grep -v __init__ | grep -v external
