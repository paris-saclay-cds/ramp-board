import shutil

import pytest

from ramputils import generate_flask_config
from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Model
from rampdb.testing import create_toy_db
from rampdb.utils import setup_db
from rampdb.utils import session_scope

from rampdb.tools.user import add_user

from frontend import create_app
from frontend.testing import login
from frontend.testing import logout


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture
def client_session(config):
    try:
        create_toy_db(config)
        flask_config = generate_flask_config(config)
        app = create_app(flask_config)
        with session_scope(config['sqlalchemy']) as session:
            yield app.test_client(), session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


# def test_approve_users(client_session):
#     client, session = client_session
#     add_user(
#         session, name='new_user', password='xxx', lastname='xxx',
#         firstname='xxx', email='new_user@xx.com',
#     )
#     rv = client.get('/approve_users', follow_redirects=True)
#     assert b'Login Error' in rv.data
#     assert b'Please log in or sign up to access this page' in rv.data
#     login(client, 'test_iris_admin', 'test')
#     rv = client.get('/approve_users', follow_redirects=True)
#     print(rv.data)
