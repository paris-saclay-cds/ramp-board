import datetime
import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Model

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.event import add_event_admin
from ramp_database.tools.event import get_event
from ramp_database.tools.event import get_event_admin
from ramp_database.tools.user import add_user
from ramp_database.tools.user import get_user_by_name

from ramp_database.tools.frontend import is_admin
from ramp_database.tools.frontend import is_accessible_code
from ramp_database.tools.frontend import is_accessible_event
from ramp_database.tools.frontend import is_accessible_leaderboard
from ramp_database.tools.frontend import is_user_signed_up


@pytest.fixture(scope='module')
def session_toy_db():
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    try:
        create_toy_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(
            ramp_config['ramp']['deployment_dir'], ignore_errors=True
        )
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_check_admin(session_toy_db):
    event_name = 'iris_test'
    user_name = 'test_iris_admin'
    assert is_admin(session_toy_db, event_name, user_name)
    user_name = 'test_user'
    assert not is_admin(session_toy_db, event_name, user_name)
    add_event_admin(session_toy_db, event_name, user_name)
    assert is_admin(session_toy_db, event_name, user_name)
    event_admin = get_event_admin(session_toy_db, event_name, user_name)
    assert event_admin.event.name == event_name
    assert event_admin.admin.name == user_name
    user_name = 'test_user_2'
    assert get_event_admin(session_toy_db, event_name, user_name) is None


@pytest.mark.parametrize(
    "event_name, user_name, is_accessible",
    [('xxx', 'test_iris_admin', False),
     ('iris_test', 'test_user', False),
     ('iris_test', 'test_iris_admin', True),
     ('iris_test', 'test_user_2', True),
     ('boston_housing_test', 'test_user_2', False)]
)
def test_is_accessible_event(session_toy_db, event_name, user_name,
                             is_accessible):
    # force one of the user to not be approved
    if user_name == 'test_user':
        user = get_user_by_name(session_toy_db, user_name)
        user.access_level = 'asked'
        session_toy_db.commit()
    # force an event to be private
    if event_name == 'boston_housing_test':
        event = get_event(session_toy_db, event_name)
        event.is_public = False
        session_toy_db.commit()
    assert is_accessible_event(session_toy_db, event_name,
                               user_name) is is_accessible


@pytest.mark.parametrize(
    "event_name, user_name, is_accessible",
    [('boston_housing', 'test_iris_admin', False),
     ('boston_housing_test', 'test_iris_admin', False),
     ('iris_test', 'test_user', True)]
)
def test_user_signed_up(session_toy_db, event_name, user_name, is_accessible):
    assert is_user_signed_up(session_toy_db, event_name,
                             user_name) is is_accessible


def test_is_accessible_code(session_toy_db):
    event_name = 'iris_test'
    # simulate a user which is not authenticated
    user = get_user_by_name(session_toy_db, 'test_user_2')
    user.is_authenticated = False
    assert not is_accessible_code(session_toy_db, event_name, user.name)
    # simulate a user which authenticated and author of the submission to a
    # public event
    user.is_authenticated = True
    assert is_accessible_code(session_toy_db, event_name, user.name)
    # simulate an admin user
    user = get_user_by_name(session_toy_db, 'test_iris_admin')
    user.is_authenticated = True
    assert is_accessible_code(session_toy_db, event_name, 'test_iris_admin')
    # simulate a user which is not signed up to the event
    user = add_user(session_toy_db, 'xx', 'xx', 'xx', 'xx', 'xx', 'user')
    user.is_authenticated = True
    assert not is_accessible_code(session_toy_db, event_name, user.name)


def test_is_accessible_leaderboard(session_toy_db):
    event_name = 'iris_test'
    # simulate a user which is not authenticated
    user = get_user_by_name(session_toy_db, 'test_user_2')
    user.is_authenticated = False
    assert not is_accessible_leaderboard(session_toy_db, event_name, user.name)
    # simulate a user which authenticated and author of the submission to a
    # public event
    user.is_authenticated = True
    assert is_accessible_leaderboard(session_toy_db, event_name, user.name)
    # simulate an admin user
    user = get_user_by_name(session_toy_db, 'test_iris_admin')
    user.is_authenticated = True
    assert is_accessible_leaderboard(session_toy_db, event_name,
                                     'test_iris_admin')
    # simulate a close event
    event = get_event(session_toy_db, event_name)
    event.closing_timestamp = datetime.datetime.utcnow()
    assert not is_accessible_leaderboard(session_toy_db, event_name,
                                         'test_user_2')
