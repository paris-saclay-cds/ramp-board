import shutil

import pytest

from ramputils import read_config
from rampdb.utils import setup_db
from ramputils.testing import path_config_example

from rampdb.model import Model

from rampdb.tools.deploy import create_test_db

from rampdb.tools.deploy import add_users
from rampdb.tools.deploy import add_problems


@pytest.fixture
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')

@pytest.fixture
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


def test_add_users(db_function, config):
    add_users(config)
    # users = db.session.query(User).all()
    # for user in users:
    #     assert user.name in ('test_user', 'test_user_2', 'test_iris_admin')
    # err_msg = 'username is already in use'
    # with pytest.raises(NameClashError, match=err_msg):
    #     add_users()


def test_add_problems(db_function, database_config, config):
    add_problems(config)
#     problems = db.session.query(Problem).all()
#     for problem in problems:
#         assert problem.name in ('iris', 'boston_housing')
#     # trying to add twice the same problem will raise a git error since the
#     #  repositories already exist.
#     with pytest.raises(GitCommandError):
#         add_problems()