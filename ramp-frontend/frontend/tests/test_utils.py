import shutil

import pytest

from ramputils import generate_flask_config
from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Model
from rampdb.testing import create_toy_db
from rampdb.utils import setup_db
from rampdb.utils import session_scope

from rampdb.tools.user import get_user_by_name

from frontend import create_app
from frontend import mail
from frontend.utils import body_formatter_user
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


@pytest.mark.xfail
def test_send_mail(client_session):
    client, _ = client_session
    with client.application.app_context():
        with mail.record_messages() as outbox:
            send_mail('xx@gmail.com', 'subject', 'body')
            assert len(outbox) == 1
            assert outbox[0].subject == 'subject'
            assert outbox[0].body == 'body'
            assert outbox[0].recipients == ['xx@gmail.com']


def test_body_formatter_user(client_session):
    _, session = client_session
    user = get_user_by_name(session, 'test_user')
    expected_body = """
    user = b'test_user'
    name = b'User' b'Test'
    email = test.user@gmail.com
    linkedin = 
    twitter = 
    facebook = 
    github = 
    notes = b''
    bio = b''

    """
    assert body_formatter_user(user) == expected_body
