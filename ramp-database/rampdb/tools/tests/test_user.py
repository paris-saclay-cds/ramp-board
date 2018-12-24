import shutil

import pytest

from ramputils import read_config
from ramputils.password import check_password
from ramputils.testing import path_config_example

from rampdb.utils import setup_db
from rampdb.model import Model
from rampdb.tools.testing import create_test_db

from rampdb.tools.user import create_user

from rampdb.tools.user import get_team_by_name
from rampdb.tools.user import get_user_by_name


@pytest.fixture(scope='module')
def config_database():
    return read_config(path_config_example(), filter_section='sqlalchemy')

@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture
def db_function(config):
    try:
        create_test_db(config)
        yield
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


def test_create_user(config_database, db_function):
    name = 'test_user'
    password = 'test'
    lastname = 'Test'
    firstname = 'User'
    email = 'test.user@gmail.com'
    access_level = 'asked'
    create_user(config_database, name=name, password=password,
                lastname=lastname, firstname=firstname, email=email,
                access_level=access_level)
    user = get_user_by_name(config_database, name)
    assert user.name == name
    assert check_password(password, user.hashed_password)
    assert user.lastname == lastname
    assert user.firstname == firstname
    assert user.email == email
    assert user.access_level == access_level
    # check that a team was automatically added with the new user
    team = get_team_by_name(config_database, name)
    assert team.name == name
    assert team.admin_id == user.id
