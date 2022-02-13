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

from ramp_database.tools.user import get_user_by_name

from ramp_frontend import create_app
from ramp_frontend.utils import body_formatter_user


@pytest.fixture(scope="module")
def client_session(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        flask_config = generate_flask_config(database_config)
        app = create_app(flask_config)
        app.config["TESTING"] = True
        app.config["WTF_CSRF_ENABLED"] = False
        with session_scope(database_config["sqlalchemy"]) as session:
            yield app.test_client(), session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        try:
            # In case of failure we should close the global flask engine
            from ramp_frontend import db as db_flask

            db_flask.session.close()
        except RuntimeError:
            pass
        db, _ = setup_db(database_config["sqlalchemy"])
        Model.metadata.drop_all(db)


def test_body_formatter_user(client_session):
    _, session = client_session
    user = get_user_by_name(session, "test_user")
    for word in [
        "test_user",
        "User",
        "Test",
        "linkedin",
        "twitter",
        "facebook",
        "github",
        "notes",
        "bio",
    ]:
        assert word in body_formatter_user(user)
