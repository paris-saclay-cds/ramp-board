import shutil

from flask import appcontext_pushed

import pytest

from ramputils import generate_flask_config
from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Model
from rampdb.testing import create_toy_db
from rampdb.utils import setup_db
from rampdb.utils import session_scope

from frontend import create_app
from frontend.utils import send_mail


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
            db_flask.session.close()
        except RuntimeError:
            pass
        db, _ = setup_db(config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_send_mail(client_session):
    client, _ = client_session
    with client.application.app_context():
        send_mail('xx@gmail.com', 'xxx', 'xxxx')
