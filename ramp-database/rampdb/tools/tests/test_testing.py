import shutil

import pytest
from git.exc import GitCommandError

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.utils import setup_db
from rampdb.utils import session_scope

from rampdb.model import Model

from rampdb.exceptions import NameClashError

from rampdb.tools.user import get_user_by_name
from rampdb.tools.event import get_problem

from rampdb.tools.testing import create_test_db
from rampdb.tools.testing import add_users
from rampdb.tools.testing import add_problems


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


def test_add_users(session_scope_function):
    add_users(session_scope_function)
    users = get_user_by_name(session_scope_function, None)
    for user in users:
        assert user.name in ('test_user', 'test_user_2', 'test_iris_admin')
    err_msg = 'username is already in use'
    with pytest.raises(NameClashError, match=err_msg):
        add_users(session_scope_function)


def test_add_problems(session_scope_function, config):
    add_problems(session_scope_function, config)
    problems = get_problem(session_scope_function, None)
    for problem in problems:
        assert problem.name in ('iris', 'boston_housing')
    # trying to add twice the same problem will raise a git error since the
    #  repositories already exist.
    with pytest.raises(GitCommandError):
        add_problems(session_scope_function, config)
