import shutil

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Model
from rampdb.model import ScoreType

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture(scope='module')
def session_scope_module(config):
    try:
        create_toy_db(config)
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


def test_score_type_model(session_scope_module):
    score_type = session_scope_module.query(ScoreType).first()
    assert 'ScoreType(name=' in repr(score_type)
