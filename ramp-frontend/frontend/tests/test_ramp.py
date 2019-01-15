import shutil

import pytest

from ramputils import generate_flask_config
from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Model
from rampdb.testing import create_toy_db
from rampdb.utils import setup_db
from rampdb.utils import session_scope

from frontend import create_app
from frontend.testing import login_scope


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture(scope='module')
def client_session(config):
    try:
        create_toy_db(config)
        flask_config = generate_flask_config(config)
        app = create_app(flask_config)
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with session_scope(config['sqlalchemy']) as session:
            yield app.test_client(), session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        try:
            # In case of failure we should close the global flask engine
            from frontend import db as db_flask
            db_flask.close()
        except Exception:
            pass
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


def test_problems(client_session):
    client, session = client_session

    # GET: access the problems page without login
    rv = client.get('/problems')
    assert rv.status_code == 200
    assert b'Hi User!' not in rv.data
    assert b'number of participants =' in rv.data
    assert b'Iris classification' in rv.data
    assert b'Boston housing price regression' in rv.data

    # GET: access the problems when logged-in
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('/problems')
        assert rv.status_code == 200
        assert b'Hi User!' in rv.data
        assert b'number of participants =' in rv.data
        assert b'Iris classification' in rv.data
        assert b'Boston housing price regression' in rv.data


def test_problem(client_session):
    client, session = client_session

    # Access a problem that does not exist
    rv = client.get('/problems/xxx')
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/problems'
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert flash_message['message'] == "Problem xxx does not exist"
    rv = client.get('/problems/xxx', follow_redirects=True)
    assert rv.status_code == 200
    print(rv.data)

