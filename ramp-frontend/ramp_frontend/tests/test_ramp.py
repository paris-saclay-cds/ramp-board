import os
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

from ramp_database.tools.event import get_event
from ramp_database.tools.user import add_user
from ramp_database.tools.submission import get_submission_by_name
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
    ["/events/iris_test",
     "/events/iris_test/sign_up",
     "/events/iris_test/sandbox",
     "problems/iris/ask_for_event",
     "/credit/xxx",
     "/event_plots/iris_test"]
)
def test_check_login_required(client_session, page):
    client, _ = client_session

    rv = client.get(page)
    assert rv.status_code == 302
    assert 'http://localhost/login' in rv.location
    rv = client.get(page, follow_redirects=True)
    assert rv.status_code == 200


@pytest.mark.parametrize(
    "page",
    ["/events/xxx",
     "/events/xxx/sign_up",
     "/events/xxx/sandbox",
     "/event_plots/xxx"]
)
def test_check_unknown_events(client_session, page):
    client, _ = client_session

    # trigger that the event does not exist
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get(page)
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "no event named" in flash_message['message']


def test_problems(client_session):
    client, _ = client_session

    # GET: access the problems page without login
    rv = client.get('/problems')
    assert rv.status_code == 200
    assert b'Hi User!' not in rv.data
    assert b'number of participants =' in rv.data
    assert b'Iris classification' in rv.data
    assert b'Boston housing price regression' in rv.data

    # GET: access the problems when logged-in
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('/problems')
        assert rv.status_code == 200
        assert b'Hi User!' in rv.data
        assert b'number of participants =' in rv.data
        assert b'Iris classification' in rv.data
        assert b'Boston housing price regression' in rv.data


def test_problem(client_session):
    client, session = client_session

    # Access a problem that does not exist
    rv = client.get('/problems/xxx')
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/problems'
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert flash_message['message'] == "Problem xxx does not exist"
    rv = client.get('/problems/xxx', follow_redirects=True)
    assert rv.status_code == 200

    # GET: looking at the problem without being logged-in
    rv = client.get('problems/iris')
    assert rv.status_code == 200
    assert b'Iris classification' in rv.data
    assert b'Current events on this problem' in rv.data
    assert b'Keywords' in rv.data

    # GET: looking at the problem being logged-in
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('problems/iris')
        assert rv.status_code == 200
        assert b'Iris classification' in rv.data
        assert b'Current events on this problem' in rv.data
        assert b'Keywords' in rv.data


def test_user_event(client_session):
    client, session = client_session

    # behavior when a user is not approved yet
    add_user(session, 'xx', 'xx', 'xx', 'xx', 'xx', access_level='asked')
    with login_scope(client, 'xx', 'xx') as client:
        rv = client.get('/events/iris_test')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert (flash_message['message'] ==
                "Your account has not been approved yet by the administrator")

    # trigger that the event does not exist
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('/events/xxx')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "no event named" in flash_message['message']

    # GET
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('events/iris_test')
        assert rv.status_code == 200
        assert b'Iris classification' in rv.data
        assert b'Rules' in rv.data
        assert b'RAMP on iris' in rv.data


def test_sign_up_for_event(client_session):
    client, session = client_session

    # trigger that the event does not exist
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('/events/xxx/sign_up')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "no event named" in flash_message['message']

    # GET: sign-up to a new controlled event
    add_user(session, 'yy', 'yy', 'yy', 'yy', 'yy', access_level='user')
    with login_scope(client, 'yy', 'yy') as client:
        rv = client.get('/events/iris_test/sign_up')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "Sign-up request is sent" in flash_message['Request sent']
        # make sure that the database has been updated for our session
        session.commit()
        event_team = get_event_team_by_name(session, 'iris_test', 'yy')
        assert not event_team.approved

    # GET: sign-up to a new uncontrolled event
    event = get_event(session, 'boston_housing_test')
    event.is_controled_signup = False
    session.commit()
    with login_scope(client, 'yy', 'yy') as client:
        rv = client.get('/events/boston_housing_test/sign_up')
        assert rv.status_code == 302
        assert (rv.location ==
                'http://localhost/events/boston_housing_test/sandbox')
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "is signed up for" in flash_message['Successful sign-up']
        # make sure that the database has been updated for our session
        session.commit()
        event_team = get_event_team_by_name(session, 'boston_housing_test',
                                            'yy')
        assert event_team.approved


def test_ask_for_event(client_session):
    client, session = client_session

    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('/problems/xxx/ask_for_event')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "no problem named" in flash_message['message']

        rv = client.get('problems/iris/ask_for_event')
        assert rv.status_code == 200
        assert b'Ask for a new event on iris' in rv.data

        data = {
            'suffix': 'test_2',
            'title': 'whatever title',
            'n_students': 200,
            'min_duration_between_submissions_hour': 1,
            'min_duration_between_submissions_minute': 2,
            'min_duration_between_submissions_second': 3,
            'opening_date': '2019-01-01',
            'closing_date': '2020-01-01'
        }
        rv = client.post('problems/iris/ask_for_event', data=data)
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert ("Thank you. Your request has been sent" in
                flash_message['Event request'])


# TODO: to be tested
# def test_sandbox(client_session):
#     client, session = client_session


# TODO: required to have run some submission
# def test_event_plots(client_session):
#     client, session = client_session


# TODO: test the behavior with a non code file
# TODO: test the importing behavior
def test_view_model(client_session):
    client, session = client_session

    # unknown submission
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('/xxxxx/xx.py')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "Missing submission" in flash_message['message']

    submission = get_submission_by_name(session, 'iris_test', 'test_user',
                                        'random_forest_10_10')
    submission_hash = submission.hash_

    # unknown workflow element
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('{}/{}'.format(submission_hash, 'extractor.py'))
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "is not a valid workflow element" in flash_message['message']

    # The file does not exist on the server
    # temporary rename the file
    os.rename(submission.path, submission.path + 'xxxxx')
    try:
        with login_scope(client, 'test_user', 'test') as client:
            rv = client.get('{}/{}'.format(submission_hash, 'classifier.py'))
            assert rv.status_code == 302
            assert rv.location == 'http://localhost/problems'
            with client.session_transaction() as cs:
                flash_message = dict(cs['_flashes'])
            assert "does not exist by" in flash_message['message']
    finally:
        os.rename(submission.path + 'xxxxx', submission.path)

    # GET: normal file display
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('{}/{}'.format(submission_hash, 'classifier.py'))
        assert rv.status_code == 200
        assert b'file = classifier.py' in rv.data
        assert b'from sklearn.base import BaseEstimator' in rv.data


def test_view_submission_error(client_session):
    client, session = client_session

    # unknown submission
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('/xxxxx/error.txt')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "Missing submission" in flash_message['message']

    submission = get_submission_by_name(session, 'iris_test', 'test_user',
                                        'error')
    submission.error_msg = 'This submission is a failure'
    session.commit()
    submission_hash = submission.hash_
    # GET: normal error display
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('{}/{}'.format(submission_hash, 'error.txt'))
        assert rv.status_code == 200
        assert b'This submission is a failure' in rv.data
