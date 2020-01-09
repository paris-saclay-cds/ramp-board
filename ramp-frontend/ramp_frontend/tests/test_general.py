import shutil

import pytest

from ramp_utils import generate_flask_config
from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Model
from ramp_database.testing import create_toy_db
from ramp_database.utils import setup_db
from ramp_database.utils import session_scope

from ramp_frontend import create_app


@pytest.fixture(scope='module')
def client_session(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        flask_config = generate_flask_config(database_config)
        app = create_app(flask_config)
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with session_scope(database_config['sqlalchemy']) as session:
            yield app.test_client(), session
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


def test_index(client_session):
    client, _ = client_session
    rv = client.get('/')
    assert rv.status_code == 200
    assert (b'RAMP: collaborative data science challenges' in
            rv.data)


def test_ramp(client_session):
    client, _ = client_session
    rv = client.get('/description')
    assert rv.status_code == 200
    assert (b'The RAMP software packages were developed by the' in
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
    assert b'RAMP challenges begin with an interesting supervised prediction' \
        in rv.data


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
