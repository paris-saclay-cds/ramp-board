import re
import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import EventScoreType
from ramp_database.model import Model
from ramp_database.model import ScoreType

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db


@pytest.fixture(scope='module')
def session_scope_module(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_score_type_model(session_scope_module):
    score_type = session_scope_module.query(ScoreType).first()
    assert re.match(r'ScoreType\(name=.*\)', repr(score_type))


@pytest.mark.parametrize(
    'backref, expected_type',
    [('events', EventScoreType)]
)
def test_score_type_model_backref(session_scope_module, backref,
                                  expected_type):
    score_type = session_scope_module.query(ScoreType).first()
    backref_attr = getattr(score_type, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)
