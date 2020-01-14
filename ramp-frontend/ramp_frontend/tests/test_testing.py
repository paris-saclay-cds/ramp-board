import shutil

import pytest

from ramp_utils import generate_flask_config
from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Model
from ramp_database.testing import create_toy_db
from ramp_database.utils import setup_db

from ramp_frontend import create_app
from ramp_frontend.testing import login
from ramp_frontend.testing import logout


@pytest.fixture
def client(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        flask_config = generate_flask_config(database_config)
        app = create_app(flask_config)
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        yield app.test_client()
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        try:
            # In case of failure we should close the global flask engine
            from ramp_frontend import db as db_flask
            db_flask.session.close()
        except RuntimeError:
            pass
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_login_logout(client):
    # check that we can login and logout a client
    rv = login(client, 'test_iris_admin', 'test')
    # the login function will follow the redirection and should be a 200 code
    assert rv.status_code == 200
    assert b'iris.admin@gmail.com' in rv.data
    assert b'Iris classification' in rv.data
    assert b'Boston housing price regression' in rv.data
    rv = logout(client)
    assert rv.status_code == 200
    assert b'Login' in rv.data
    assert b'Username' in rv.data
    assert b'Password' in rv.data
