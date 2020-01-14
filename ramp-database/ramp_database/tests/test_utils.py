import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.testing import create_test_db

from ramp_database.model import Model
from ramp_database.model import SubmissionFileType

from ramp_database.utils import check_password
from ramp_database.utils import hash_password
from ramp_database.utils import setup_db
from ramp_database.utils import session_scope


@pytest.fixture
def database(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_test_db(database_config, ramp_config)
        yield
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_setup_db(database):
    database_config = read_config(
        database_config_template(), filter_section='sqlalchemy'
    )
    db, Session = setup_db(database_config)
    with db.connect() as conn:
        session = Session(bind=conn)
        file_type = session.query(SubmissionFileType).all()
        assert len(file_type) > 0


def test_session_scope(database):
    database_config = read_config(
        database_config_template(), filter_section='sqlalchemy'
    )
    with session_scope(database_config) as session:
        file_type = session.query(SubmissionFileType).all()
        assert len(file_type) > 0


def test_check_password():
    password = "hjst3789ep;ocikaqjw"
    hashed_password = hash_password(password)
    assert check_password(password, hashed_password)
    assert not check_password("hjst3789ep;ocikaqji", hashed_password)
