import shutil

import pytest

from ramputils import generate_flask_config
from ramputils import read_config
from ramputils.testing import database_config_template
from ramputils.testing import ramp_config_template

from rampdb.model import Model
from rampdb.testing import create_toy_db
from rampdb.utils import setup_db
from rampdb.utils import session_scope

from frontend import create_app


@pytest.fixture(scope='module')
def client_session():
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    try:
        create_toy_db(database_config, ramp_config)
        flask_config = generate_flask_config(database_config)
        app = create_app(flask_config)
        app.config['TESTING'] = True
        app.config['WTF_CSRF_ENABLED'] = False
        with session_scope(database_config['sqlalchemy']) as session:
            yield app.test_client(), session
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
