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

from rampdb.tools.user import get_user_by_name

from frontend import create_app
from frontend import mail
from frontend.utils import body_formatter_user
from frontend.utils import send_mail


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
    for word in ['test_user', 'User', 'Test', 'linkedin', 'twitter',
                 'facebook', 'github', 'notes', 'bio']:
        assert word in body_formatter_user(user)
