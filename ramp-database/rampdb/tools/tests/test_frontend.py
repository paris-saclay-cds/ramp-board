import shutil

import pytest

from ramputils import read_config
from ramputils import generate_ramp_config
from ramputils.testing import path_config_example

from rampdb.model import Model

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.event import add_event_admin
from rampdb.tools.event import get_event
from rampdb.tools.event import get_event_admin
from rampdb.tools.user import get_user_by_name

from rampdb.tools.frontend import is_admin
from rampdb.tools.frontend import is_accessible_code
from rampdb.tools.frontend import is_accessible_event
from rampdb.tools.frontend import is_accessible_leaderboard
from rampdb.tools.frontend import is_user_signed_up


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture(scope='module')
def session_toy_db(config):
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
