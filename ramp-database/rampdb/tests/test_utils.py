import shutil

import pytest

from ramputils import read_config
from ramputils.testing import database_config_template
from ramputils.testing import ramp_config_template

from rampdb.testing import create_test_db

from rampdb.model import Model
from rampdb.model import SubmissionFileType

from rampdb.utils import setup_db
from rampdb.utils import session_scope

@pytest.fixture
def database():
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    try:
        create_test_db(database_config, ramp_config)
        yield
    finally:
        shutil.rmtree(
            ramp_config['ramp']['deployment_dir'], ignore_errors=True
        )
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
