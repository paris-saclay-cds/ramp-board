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

from rampdb.tools.event import get_event
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

    # ensure that the previous change have been committed within our session
    session.commit()
    user = get_user_by_name(session, 'yy')
    assert user.access_level == 'user'
    event_team = get_event_team_by_name(session, 'iris_test', 'xx')
    assert event_team.approved
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert re.match(r"Approved users:\nyy\nApproved event_team:\n"
                    r"Event\(iris_test\)/Team\(.*xx.*\)\n",
                    flash_message['Approved users'])

    logout(client)


def test_approve_sign_up_for_event(client_session):
    client, session = client_session

    login(client, 'test_iris_admin', 'test')

    # check the redirection if the user or the event does not exist
    rv = client.get("/events/xxx/sign_up/test_user")
    session.commit()
    assert rv.status_code == 302
    assert rv.location == "http://localhost/problems"
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert flash_message['message'] == 'No event xxx or no user test_user'

    rv = client.get("/events/iris_test/sign_up/xxxx")
    session.commit()
    assert rv.status_code == 302
    assert rv.location == "http://localhost/problems"
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert flash_message['message'] == 'No event iris_test or no user xxxx'

    add_user(session, 'zz', 'zz', 'zz', 'zz', 'zz', access_level='user')
    _, _, event_team = ask_sign_up_team(session, 'iris_test', 'zz')
    assert not event_team.approved
    rv = client.get('/events/iris_test/sign_up/zz')
    assert rv.status_code == 302
    assert rv.location == "http://localhost/problems"
    session.commit()
    event_team = get_event_team_by_name(session, 'iris_test', 'zz')
    assert event_team.approved
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert "is signed up for Event" in flash_message['Successful sign-up']

    logout(client)


def test_update_event(client_session):
    client, session = client_session

    login(client, 'test_iris_admin', 'test')

    # case tha the event does not exist
    rv = client.get('/events/boston_housing/update')
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/problems'
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert 'no event named "boston_housing"' in flash_message['message']

    # GET: pre-fill the forms
    rv = client.get('/events/iris_test/update')
    assert rv.status_code == 200
    assert b'Minimum duration between submissions' in rv.data

    # POST: update the event data
    event_info = {
        'suffix': 'test',
        'title': 'Iris new title',
        'is_send_trained_mail': True,
        'is_public': True,
        'is_controled_signup': True,
        'is_competitive': False,
        'min_duration_between_submissions_hour': 0,
        'min_duration_between_submissions_minute': 0,
        'min_duration_between_submissions_second': 0,
        'opening_timestamp': "2000-01-01 00:00:00",
        'closing_timestamp': "2100-01-01 00:00:00",
        'public_opening_timestamp': "2000-01-01 00:00:00"
    }
    rv = client.post('/events/iris_test/update', data=event_info)
    assert rv.status_code == 302
    assert rv.location == "http://localhost/problems"

    logout(client)
