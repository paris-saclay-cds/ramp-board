import shutil

import pytest

from ramputils import generate_flask_config
from ramputils import read_config
from ramputils.testing import database_config_template
from ramputils.testing import ramp_config_template

from rampdb.model import Model
from rampdb.testing import create_toy_db
from rampdb.utils import setup_db

from frontend import create_app
from frontend.testing import login
from frontend.testing import logout


@pytest.fixture
def client():
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    try:
        create_toy_db(database_config, ramp_config)
        flask_config = generate_flask_config(database_config)
        app = create_app(flask_config)
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        yield app.test_client()
    finally:
        shutil.rmtree(
            ramp_config['ramp']['deployment_dir'], ignore_errors=True
        )
        try:
            # In case of failure we should close the global flask engine
            from frontend import db as db_flask
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
