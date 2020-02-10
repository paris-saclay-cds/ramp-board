import re
import shutil

import pytest

from werkzeug.datastructures import ImmutableMultiDict

from ramp_utils import generate_flask_config
from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Model
from ramp_database.testing import create_toy_db
from ramp_database.utils import setup_db
from ramp_database.utils import session_scope

from ramp_database.tools.event import get_event
from ramp_database.tools.user import add_user
from ramp_database.tools.user import get_user_by_name
from ramp_database.tools.team import ask_sign_up_team
from ramp_database.tools.team import get_event_team_by_name

from ramp_frontend import create_app
from ramp_frontend.testing import login_scope


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


@pytest.mark.parametrize(
    "page",
    ["/approve_users",
     "/sign_up/test_user",
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
     ("/sign_up/test_user", ["get"]),
     ("/events/iris_test/sign_up/test_user", ["get"]),
     ("/events/iris_test/update", ["get", "post"]),
     ("/user_interactions", ["get"]),
     ("/events/iris_test/dashboard_submissions", ["get"])]
)
def test_check_admin_required(client_session, page, request_function):
    client, _ = client_session

    with login_scope(client, 'test_user', 'test') as client:
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


def test_approve_users_remove(client_session):
    client, session = client_session

    # create 2 new users
    add_user(session, 'xx', 'xx', 'xx', 'xx', 'xx', access_level='user')
    add_user(session, 'yy', 'yy', 'yy', 'yy', 'yy', access_level='asked')
    # ask for sign up for an event for the first user
    _, _, event_team = ask_sign_up_team(session, 'iris_test', 'xx')

    with login_scope(client, 'test_iris_admin', 'test') as client:

        # GET check that we get all new user to be approved
        rv = client.get('/approve_users')
        assert rv.status_code == 200
        # line for user approval
        assert b'yy yy - yy' in rv.data
        # line for the event approval
        assert b'iris_test - xx'

        # POST check that we are able to approve a user and event
        data = ImmutableMultiDict([
            ('submit_button', 'Remove!'),
            ('approve_users', 'yy'),
            ('approve_event_teams', str(event_team.id))
        ])
        rv = client.post('/approve_users', data=data)
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'

        # ensure that the previous change have been committed within our
        # session
        session.commit()
        user = get_user_by_name(session, 'yy')
        assert user is None
        event_team = get_event_team_by_name(session, 'iris_test', 'xx')
        assert event_team is None
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert re.match(r"Removed users:\nyy\nRemoved event_team:\n"
                        r"Event\(iris_test\)/Team\(.*xx.*\)\n",
                        flash_message['Removed users'])


def test_approve_users_approve(client_session):
    client, session = client_session

    # create 2 new users
    add_user(session, 'cc', 'cc', 'cc', 'cc', 'cc', access_level='user')
    add_user(session, 'dd', 'dd', 'dd', 'dd', 'dd', access_level='asked')
    # ask for sign up for an event for the first user
    _, _, event_team = ask_sign_up_team(session, 'iris_test', 'cc')

    with login_scope(client, 'test_iris_admin', 'test') as client:

        # GET check that we get all new user to be approved
        rv = client.get('/approve_users')
        assert rv.status_code == 200
        # line for user approval
        assert b'dd dd - dd' in rv.data
        # line for the event approval
        assert b'iris_test - cc'

        # POST check that we are able to approve a user and event
        data = ImmutableMultiDict([
            ('submit_button', 'Approve!'),
            ('approve_users', 'dd'),
            ('approve_event_teams', str(event_team.id))]
        )
        rv = client.post('/approve_users', data=data)
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'

        # ensure that the previous change have been committed within our
        # session
        session.commit()
        user = get_user_by_name(session, 'dd')
        assert user.access_level == 'user'
        event_team = get_event_team_by_name(session, 'iris_test', 'cc')
        assert event_team.approved
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert re.match(r"Approved users:\ndd\nApproved event_team:\n"
                        r"Event\(iris_test\)/Team\(.*cc.*\)\n",
                        flash_message['Approved users'])


def test_approve_single_user(client_session):
    client, session = client_session

    add_user(session, 'gg', 'gg', 'gg', 'gg', 'gg', access_level='asked')
    with login_scope(client, 'test_iris_admin', 'test') as client:
        rv = client.get('/sign_up/gg')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert re.match("User(.*gg.*) is signed up",
                        flash_message['Successful sign-up'])

        # ensure that the previous change have been committed within our
        # session
        session.commit()
        user = get_user_by_name(session, 'gg')
        assert user.access_level == 'user'

        rv = client.get("/sign_up/unknown_user")
        session.commit()
        assert rv.status_code == 302
        assert rv.location == "http://localhost/problems"
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert flash_message['message'] == 'No user unknown_user'


def test_approve_sign_up_for_event(client_session):
    client, session = client_session

    with login_scope(client, 'test_iris_admin', 'test') as client:

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


def test_update_event(client_session):
    client, session = client_session

    with login_scope(client, 'test_iris_admin', 'test') as client:

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
            'public_opening_timestamp': "2000-01-01 00:00:00",
        }
        rv = client.post('/events/iris_test/update', data=event_info)
        assert rv.status_code == 302
        assert rv.location == "http://localhost/problems"
        event = get_event(session, 'iris_test')
        assert event.min_duration_between_submissions == 0


def test_user_interactions(client_session):
    client, _ = client_session

    with login_scope(client, 'test_iris_admin', 'test') as client:
        rv = client.get('/user_interactions')
        assert rv.status_code == 200
        assert b'landing' in rv.data


# TODO: To be tested when we implemented properly the leaderboard
# def test_dashboard_submissions(client_session):
#     client, session = client_session

#     with login_scope(client, 'test_iris_admin', 'test') as client:
#         rv = client.get('/events/iris_test/dashboard_submissions')
#         print(rv.data.decode('utf-8'))
