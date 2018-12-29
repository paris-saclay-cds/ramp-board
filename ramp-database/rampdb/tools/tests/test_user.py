import shutil

import pytest

from ramputils import read_config
from ramputils.password import check_password
from ramputils.testing import path_config_example

from rampdb.utils import setup_db
from rampdb.utils import session_scope

from rampdb.model import Model
from rampdb.model import User
from rampdb.model import Team
from rampdb.testing import create_test_db

from rampdb.tools.user import approve_user
from rampdb.tools.user import create_user
from rampdb.tools.user import get_team_by_name
from rampdb.tools.user import get_user_by_name


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture
def session_scope_function(config):
    try:
        create_test_db(config)
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


def test_create_user(session_scope_function):
    name = 'test_user'
    password = 'test'
    lastname = 'Test'
    firstname = 'User'
    email = 'test.user@gmail.com'
    access_level = 'asked'
    create_user(session_scope_function, name=name, password=password,
                lastname=lastname, firstname=firstname, email=email,
                access_level=access_level)
    user = get_user_by_name(session_scope_function, name)
    assert user.name == name
    assert check_password(password, user.hashed_password)
    assert user.lastname == lastname
    assert user.firstname == firstname
    assert user.email == email
    assert user.access_level == access_level
    # check that a team was automatically added with the new user
    team = get_team_by_name(session_scope_function, name)
    assert team.name == name
    assert team.admin_id == user.id


@pytest.mark.parametrize(
    "name, query_type", [(None, list), ('test_user', User)]
)
def test_get_user_by_name(session_scope_function, name, query_type):
    create_user(session_scope_function, name='test_user', password='password',
                lastname='lastname', firstname='firstname',
                email='test_user@email.com', access_level='asked')
    create_user(session_scope_function, name='test_user_2',
                password='password', lastname='lastname',
                firstname='firstname', email='test_user_2@email.com',
                access_level='asked')
    user = get_user_by_name(session_scope_function, name)
    assert isinstance(user, query_type)


@pytest.mark.parametrize(
    "name, query_type", [(None, list), ('test_user', Team)]
)
def test_get_team_by_name(session_scope_function, name, query_type):
    create_user(session_scope_function, name='test_user', password='password',
                lastname='lastname', firstname='firstname',
                email='test_user@email.com', access_level='asked')
    create_user(session_scope_function, name='test_user_2',
                password='password', lastname='lastname',
                firstname='firstname', email='test_user_2@email.com',
                access_level='asked')
    team = get_team_by_name(session_scope_function, name)
    assert isinstance(team, query_type)


def test_approve_user(session_scope_function):
    create_user(session_scope_function, name='test_user', password='test',
                lastname='Test', firstname='User', email='test.user@gmail.com',
                access_level='asked')
    user = get_user_by_name(session_scope_function, 'test_user')
    assert user.access_level == 'asked'
    assert user.is_authenticated is False
    approve_user(session_scope_function, 'test_user')
    user = get_user_by_name(session_scope_function, 'test_user')
    assert user.access_level == 'user'
    assert user.is_authenticated is True
