import shutil

import pandas as pd
import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.utils import check_password
from ramp_database.utils import setup_db
from ramp_database.utils import session_scope

from ramp_database.exceptions import NameClashError

from ramp_database.model import Model
from ramp_database.model import User
from ramp_database.model import Team
from ramp_database.testing import create_test_db

from ramp_database.tools.user import add_user
from ramp_database.tools.user import add_user_interaction
from ramp_database.tools.user import approve_user
from ramp_database.tools.user import delete_user
from ramp_database.tools.user import get_team_by_name
from ramp_database.tools.user import get_user_by_name
from ramp_database.tools.user import get_user_interactions_by_name
from ramp_database.tools.user import make_user_admin
from ramp_database.tools.user import set_user_access_level
from ramp_database.tools.user import set_user_by_instance


@pytest.fixture
def session_scope_function(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_test_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_add_user(session_scope_function):
    name = 'test_user'
    password = 'test'
    lastname = 'Test'
    firstname = 'User'
    email = 'test.user@gmail.com'
    access_level = 'asked'
    add_user(session_scope_function, name=name, password=password,
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
    # check that we get an error if we try to add the same user
    with pytest.raises(NameClashError, match='email is already in use'):
        add_user(session_scope_function, name=name, password=password,
                 lastname=lastname, firstname=firstname, email=email,
                 access_level=access_level)
    # check that the checking is case insensitive
    with pytest.raises(NameClashError, match='email is already in use'):
        add_user(session_scope_function, name=name, password=password,
                 lastname=lastname, firstname=firstname,
                 email=email.capitalize(), access_level=access_level)
    # add a user email with some capital letters and check that only lower case
    # are stored in the database
    name = 'new_user_name'
    email = 'MixCase@mail.com'
    add_user(session_scope_function, name=name, password=password,
             lastname=lastname, firstname=firstname, email=email,
             access_level=access_level)
    user = get_user_by_name(session_scope_function, name)
    assert user.email == 'mixcase@mail.com'


def test_delete_user(session_scope_function):
    username = 'test_user'
    add_user(
        session_scope_function, name=username, password='password',
        lastname='lastname', firstname='firstname',
        email='test_user@email.com', access_level='asked')
    user = (session_scope_function.query(User)
                                  .filter(User.name == username)
                                  .all())
    assert len(user) == 1
    delete_user(session_scope_function, username)
    user = (session_scope_function.query(User)
                                  .filter(User.name == username)
                                  .one_or_none())
    assert user is None
    team = (session_scope_function.query(Team)
                                  .filter(Team.name == username)
                                  .all())
    assert len(team) == 0


def test_make_user_admin(session_scope_function):
    username = 'test_user'
    user = add_user(
        session_scope_function, name=username, password='password',
        lastname='lastname', firstname='firstname',
        email='test_user@email.com', access_level='asked')
    assert user.access_level == 'asked'
    assert user.is_authenticated is False
    make_user_admin(session_scope_function, username)
    user = get_user_by_name(session_scope_function, username)
    assert user.access_level == 'admin'
    assert user.is_authenticated is True


@pytest.mark.parametrize("access_level", ["asked", "user", "admin"])
def test_set_user_access_level(session_scope_function, access_level):
    username = 'test_user'
    user = add_user(
        session_scope_function, name=username, password='password',
        lastname='lastname', firstname='firstname',
        email='test_user@email.com', access_level='asked')
    assert user.access_level == 'asked'
    assert user.is_authenticated is False
    set_user_access_level(session_scope_function, username, access_level)
    user = get_user_by_name(session_scope_function, username)
    assert user.access_level == access_level
    assert user.is_authenticated is True


@pytest.mark.parametrize(
    "name, query_type", [(None, list), ('test_user', User)]
)
def test_get_user_by_name(session_scope_function, name, query_type):
    add_user(session_scope_function, name='test_user', password='password',
             lastname='lastname', firstname='firstname',
             email='test_user@email.com', access_level='asked')
    add_user(session_scope_function, name='test_user_2',
             password='password', lastname='lastname',
             firstname='firstname', email='test_user_2@email.com',
             access_level='asked')
    user = get_user_by_name(session_scope_function, name)
    assert isinstance(user, query_type)


def test_set_user_by_instance(session_scope_function):
    add_user(session_scope_function, name='test_user', password='password',
             lastname='lastname', firstname='firstname',
             email='test_user@email.com', access_level='asked')
    add_user(session_scope_function, name='test_user_2',
             password='password', lastname='lastname',
             firstname='firstname', email='test_user_2@email.com',
             access_level='asked')
    user = get_user_by_name(session_scope_function, 'test_user')
    set_user_by_instance(session_scope_function, user, lastname='a',
                         firstname='b', email='c', linkedin_url='d',
                         twitter_url='e', facebook_url='f', google_url='g',
                         github_url='h', website_url='i', bio='j',
                         is_want_news=False)
    user = get_user_by_name(session_scope_function, 'test_user')
    assert user.lastname == 'a'
    assert user.firstname == 'b'
    assert user.email == 'c'
    assert user.linkedin_url == 'd'
    assert user.twitter_url == 'e'
    assert user.facebook_url == 'f'
    assert user.google_url == 'g'
    assert user.github_url == 'h'
    assert user.website_url == 'i'
    assert user.bio == 'j'
    assert user.is_want_news is False


@pytest.mark.parametrize(
    "name, query_type", [(None, list), ('test_user', Team)]
)
def test_get_team_by_name(session_scope_function, name, query_type):
    add_user(session_scope_function, name='test_user', password='password',
             lastname='lastname', firstname='firstname',
             email='test_user@email.com', access_level='asked')
    add_user(session_scope_function, name='test_user_2',
             password='password', lastname='lastname',
             firstname='firstname', email='test_user_2@email.com',
             access_level='asked')
    team = get_team_by_name(session_scope_function, name)
    assert isinstance(team, query_type)


def test_approve_user(session_scope_function):
    add_user(session_scope_function, name='test_user', password='test',
             lastname='Test', firstname='User', email='test.user@gmail.com',
             access_level='asked')
    user = get_user_by_name(session_scope_function, 'test_user')
    assert user.access_level == 'asked'
    assert user.is_authenticated is False
    approve_user(session_scope_function, 'test_user')
    user = get_user_by_name(session_scope_function, 'test_user')
    assert user.access_level == 'user'
    assert user.is_authenticated is True


@pytest.mark.parametrize(
    "output_format, expected_format",
    [('dataframe', pd.DataFrame),
     ('html', str)]
)
def test_check_user_interactions(session_scope_function, output_format,
                                 expected_format):
    add_user(session_scope_function, name='test_user', password='password',
             lastname='lastname', firstname='firstname',
             email='test_user@email.com', access_level='asked')
    params = {'interaction': 'landing'}
    add_user_interaction(session_scope_function, **params)
    params = {'interaction': 'landing',
              'user': get_user_by_name(session_scope_function, 'test_user')}
    add_user_interaction(session_scope_function, **params)
    user_interaction = get_user_interactions_by_name(
        session_scope_function, output_format=output_format)
    if isinstance(user_interaction, pd.DataFrame):
        assert user_interaction.shape[0] == 2
    assert isinstance(user_interaction, expected_format)
    user_interaction = get_user_interactions_by_name(
        session_scope_function, name='test_user', output_format=output_format)
    if isinstance(user_interaction, pd.DataFrame):
        assert user_interaction.shape[0] == 1
