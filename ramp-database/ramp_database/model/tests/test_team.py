import re
import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import EventTeam
from ramp_database.model import Model

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.user import get_team_by_name


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


def test_team_model(session_scope_module):
    team = get_team_by_name(session_scope_module, 'test_user')
    assert re.match(r'Team\(name=.*test_user.*, admin_name=.*test_user.*\)',
                    repr(team))
    assert re.match(r'Team\(.*test_user.*\)', str(team))


@pytest.mark.parametrize(
    'backref, expected_type',
    [('team_events', EventTeam)]
)
def test_event_model_backref(session_scope_module, backref, expected_type):
    team = get_team_by_name(session_scope_module, 'test_user')
    backref_attr = getattr(team, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)
