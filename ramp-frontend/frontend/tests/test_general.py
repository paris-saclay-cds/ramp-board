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


def test_index(client_session):
    client, _ = client_session
    rv = client.get('/')
    assert rv.status_code == 200
    assert (b'RAMP: collaborative data science challenges at Paris Saclay' in
            rv.data)


def test_ramp(client_session):
    client, _ = client_session
    rv = client.get('/description')
    assert rv.status_code == 200
    assert (b'The RAMP is a <b>versatile management and software tool</b>' in
            rv.data)


def test_domain(client_session):
    client, session = client_session
    rv = client.get('/data_domains')
    assert rv.status_code == 200
    assert b'Scientific data domains' in rv.data
    assert b'boston_housing' in rv.data
    assert b'Boston housing price regression' in rv.data


def test_teaching(client_session):
    client, _ = client_session
    rv = client.get('/teaching')
    assert rv.status_code == 200
    assert b'RAMP for teaching support' in rv.data


def test_data_science_themes(client_session):
    client, _ = client_session
    rv = client.get('/data_science_themes')
    assert rv.status_code == 200
    assert b'boston_housing_theme' in rv.data
    assert b'iris_theme' in rv.data

def test_keywords(client_session):
    client, _ = client_session
    rv = client.get('/keywords/boston_housing')
    assert rv.status_code == 200
    assert b'Related problems' in rv.data
    assert b'boston_housing' in rv.data
    assert b'Boston housing price regression' in rv.data
