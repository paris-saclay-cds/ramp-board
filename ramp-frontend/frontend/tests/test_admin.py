import re
import shutil

import pytest

from ramputils import generate_flask_config
from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Model
from rampdb.testing import create_toy_db
from rampdb.utils import setup_db
from rampdb.utils import session_scope

from rampdb.tools.user import add_user
from rampdb.tools.user import get_user_by_name
from rampdb.tools.team import ask_sign_up_team
from rampdb.tools.team import get_event_team_by_name

from werkzeug.datastructures import ImmutableMultiDict

from frontend import create_app
from frontend.testing import login
from frontend.testing import logout


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
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


@pytest.mark.parametrize(
    "page",
    ["/approve_users",
     "/events/iris_test/sign_up/test_user",
     "/events/iris_test/update",
     "/user_interactions",
     "/events/iris_test/dashboard_submissions"]
)
def test_check_login_required(client_session, page):
    client, _ = client_session

    rv = client.get(page)
    assert rv.status_code == 302
    assert 'http://localhost/login' in rv.location
    rv = client.get(page, follow_redirects=True)
    assert rv.status_code == 200


@pytest.mark.parametrize(
    "page, request_function",
    [("/approve_users", ["get", "post"]),
     ("/events/iris_test/sign_up/test_user", ["get"]),
     ("/events/iris_test/update", ["get", "post"]),
     ("/user_interactions", ["get"]),
     ("/events/iris_test/dashboard_submissions", ["get"])]
)
def test_check_admin_required(client_session, page, request_function):
    client, _ = client_session

    login(client, 'test_user', 'test')
    for rf in request_function:
        rv = getattr(client, rf)(page)
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert (flash_message['message'] ==
                'Sorry User, you do not have admin rights')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        rv = getattr(client, rf)(page, follow_redirects=True)
        assert rv.status_code == 200
    logout(client)


def test_approve_users(client_session):
    client, session = client_session

    # create 2 new users
    add_user(session, 'xx', 'xx', 'xx', 'xx', 'xx', access_level='user')
    add_user(session, 'yy', 'yy', 'yy', 'yy', 'yy', access_level='asked')
    # ask for sign up for an event for the first user
    _, _, event_team = ask_sign_up_team(session, 'iris_test', 'xx')
    print(event_team.id)

    login(client, 'test_iris_admin', 'test')

    # GET check that we get all new user to be approved
    rv = client.get('/approve_users')
    assert rv.status_code == 200
    # line for user approval
    assert b'yy: yy yy - yy' in rv.data
    # line for the event approval
    assert b'iris_test - xx'

    # POST check that we are able to approve a user and event
    data = ImmutableMultiDict([('approve_users', 'yy'),
                               ('approve_event_teams', str(event_team.id))])
    rv = client.post('/approve_users', data=data)
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/problems'
    rv = client.post('/approve_users', data=data, follow_redirects=True)
    user = get_user_by_name(session, 'yy')
    assert user.access_level == 'user'
    # event_team = get_event_team_by_name(session, 'iris_test', 'xx')
    print(event_team.id)
    print(event_team.approved)
    assert event_team.approved
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    print(flash_message)

    logout(client)
