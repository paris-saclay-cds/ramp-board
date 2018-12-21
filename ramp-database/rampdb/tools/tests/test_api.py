import shutil
import pytest

# TODO: we temporary use the setup of databoard to create a dataset
from databoard import db
from databoard import deployment_path
from databoard.testing import create_toy_db

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.tools import get_submissions


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture(scope='module')
def setup_db():
    try:
        create_toy_db()
        yield
    finally:
        shutil.rmtree(deployment_path, ignore_errors=True)
        db.session.close()
        db.session.remove()
        db.drop_all()


def test_get_submissions(config, setup_db):
    submissions = get_submissions(config['sqlalchemy'], 'iris_test')
    print(submissions)
