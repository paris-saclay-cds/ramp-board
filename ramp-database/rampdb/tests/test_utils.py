import shutil

import pytest

# TODO: we temporary use the setup of databoard to create a dataset
from databoard import db
from databoard import deployment_path
from databoard.testing import create_test_db

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import SubmissionFileType

from rampdb.utils import setup_db
from rampdb.utils import session_scope


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture
def database(scope='module'):
    try:
        create_test_db()
        yield
    finally:
        shutil.rmtree(deployment_path, ignore_errors=True)
        db.session.close()
        db.session.remove()
        db.drop_all()


def test_setup_db(database_config, database):
    db, Session = setup_db(database_config)
    with db.connect() as conn:
        session = Session(bind=conn)
        file_type = session.query(SubmissionFileType).all()
        assert len(file_type) > 0


def test_session_scope(database_config, database):
    with session_scope(database_config) as session:
        file_type = session.query(SubmissionFileType).all()
        assert len(file_type) > 0
