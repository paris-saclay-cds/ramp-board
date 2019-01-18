import shutil

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.testing import create_test_db

from rampdb.model import Model
from rampdb.model import SubmissionFileType

from rampdb.utils import setup_db
from rampdb.utils import session_scope


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture
def database(config):
    try:
        create_test_db(config)
        yield
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, _ = setup_db(config['sqlalchemy'])
        Model.metadata.drop_all(db)


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
