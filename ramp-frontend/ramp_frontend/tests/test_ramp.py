import datetime
import os
import shutil

import pytest

from ramp_utils import generate_flask_config
from ramp_utils import generate_ramp_config
from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Model
from ramp_database.model import Event
from ramp_database.model import Submission
from ramp_database.testing import create_toy_db
from ramp_database.utils import setup_db
from ramp_database.utils import session_scope

from ramp_database.tools.event import get_event
from ramp_database.tools.user import add_user
from ramp_database.tools.user import get_user_interactions_by_name
from ramp_database.tools.submission import get_submission_by_name
from ramp_database.tools.team import get_event_team_by_name
from ramp_database.tools.event import add_event
from ramp_database.tools.event import delete_event
from ramp_database.tools.team import sign_up_team
from ramp_database.tools.team import ask_sign_up_team

from ramp_frontend import create_app
from ramp_frontend.testing import login_scope
from ramp_frontend import mail
from ramp_frontend.testing import _fail_no_smtp_server


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


@pytest.fixture(scope='function')
def makedrop_event(client_session):
    _, session = client_session
    add_event(session, 'iris', 'iris_test_4event', 'iris_test_4event',
              'starting_kit', '/tmp/databoard_test/submissions',
              is_public=True)
    yield
    delete_event(session, 'iris_test_4event')


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
    assert b'participants' in rv.data
    assert b'Iris classification' in rv.data
    assert b'Boston housing price regression' in rv.data

    # GET: access the problems when logged-in
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('/problems')
        assert rv.status_code == 200
        assert b'Hi User!' in rv.data
        assert b'participants' in rv.data
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
    assert b'Registered events' in rv.data

    # GET: looking at the problem being logged-in
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('problems/iris')
        assert rv.status_code == 200
        assert b'Iris classification' in rv.data
        assert b'Registered events' in rv.data


@pytest.mark.parametrize(
    "event_name, correct",
    [("iri_aaa", False),
     ("irisaaa", False),
     ("test_iris", False),
     ("iris_", True),
     ("iris_aaa_aaa_test", True),
     ("iris", False),
     ("iris_t", True)]
)
def test_event_name_correct(client_session, event_name, correct):
    client, session = client_session
    if not correct:
        err_msg = "The event name should start with the problem name"
        with pytest.raises(ValueError, match=err_msg):
            add_event(
                session, 'iris', event_name, 'new_event', 'starting_kit',
                '/tmp/databoard_test/submissions', is_public=True
            )
    else:
        assert add_event(session, 'iris', event_name, 'new_event',
                         'starting_kit', '/tmp/databoard_test/submissions',
                         is_public=True)


def test_user_event_status(client_session):
    client, session = client_session

    add_user(session, 'new_user', 'new_user', 'new_user',
             'new_user', 'new_user', access_level='user')
    add_event(session, 'iris', 'iris_new_event', 'new_event', 'starting_kit',
              '/tmp/databoard_test/submissions', is_public=True)

    # user signed up, not approved for the event
    ask_sign_up_team(session, 'iris_new_event', 'new_user')
    with login_scope(client, 'new_user', 'new_user') as client:
        rv = client.get('/problems')
        assert rv.status_code == 200
        assert b'user-waiting' in rv.data
        assert b'user-signed' not in rv.data

    # user signed up and approved for the event
    sign_up_team(session, 'iris_new_event', 'new_user')
    with login_scope(client, 'new_user', 'new_user') as client:
        rv = client.get('/problems')
        assert rv.status_code == 200
        assert b'user-signed' in rv.data
        assert b'user-waiting' not in rv.data


NOW = datetime.datetime.now()
testtimestamps = [
    (NOW.replace(year=NOW.year+1), NOW.replace(year=NOW.year+2),
     NOW.replace(year=NOW.year+3), b'event-close'),
    (NOW.replace(year=NOW.year-1), NOW.replace(year=NOW.year+1),
     NOW.replace(year=NOW.year+2), b'event-comp'),
    (NOW.replace(year=NOW.year-2), NOW.replace(year=NOW.year-1),
     NOW.replace(year=NOW.year+1), b'event-collab'),
    (NOW.replace(year=NOW.year-3), NOW.replace(year=NOW.year-2),
     NOW.replace(year=NOW.year-1), b'event-close'),
]


@pytest.mark.parametrize(
    "opening_date,public_date,closing_date,expected", testtimestamps
)
def test_event_status(client_session, makedrop_event,
                      opening_date, public_date,
                      closing_date, expected):
    # checks if the event status is displayed correctly
    client, session = client_session

    # change the datetime stamps for the event
    event = get_event(session, 'iris_test_4event')
    event.opening_timestamp = opening_date
    event.public_opening_timestamp = public_date
    event.closing_timestamp = closing_date
    session.commit()

    # GET: access the problems page without login
    rv = client.get('/problems')
    assert rv.status_code == 200
    event_idx = rv.data.index(b'iris_test_4event')
    event_class_idx = rv.data[:event_idx].rfind(b'<i class')
    assert expected in rv.data[event_class_idx:event_idx]

    # GET: access the problems when logged-in
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('/problems')
        assert rv.status_code == 200
        event_idx = rv.data.index(b'iris_test_4event')
        event_class_idx = rv.data[:event_idx].rfind(b'<i class')
        assert expected in rv.data[event_class_idx:event_idx]


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
        # check that we are informing the user that he has to wait for approval
        rv = client.get('/events/iris_test')
        assert rv.status_code == 200
        assert b'Waiting approval...' in rv.data

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


@_fail_no_smtp_server
def test_sign_up_for_event_mail(client_session):
    client, session = client_session

    # GET: sign-up to a new controlled event
    with client.application.app_context():
        with mail.record_messages() as outbox:
            add_user(
                session, 'zz', 'zz', 'zz', 'zz', 'zz@gmail',
                access_level='user'
            )
            with login_scope(client, 'zz', 'zz') as client:
                rv = client.get('/events/iris_test/sign_up')
                assert rv.status_code == 302
                session.commit()
                # check that the email has been sent
                assert len(outbox) == 1
                assert ('Click on this link to approve the sign-up request'
                        in outbox[0].body)


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


@_fail_no_smtp_server
def test_ask_for_event_mail(client_session):
    client, session = client_session

    with client.application.app_context():
        with mail.record_messages() as outbox:
            with login_scope(client, 'test_user', 'test') as client:

                rv = client.get('problems/iris/ask_for_event')
                assert rv.status_code == 200
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
                # check that the email has been sent
                assert len(outbox) == 1
                assert ('User test_user asked to add a new event'
                        in outbox[0].body)


@pytest.mark.parametrize(
    "opening_date, public_date, closing_date, expected", testtimestamps
)
def test_submit_button_enabled_disabled(client_session, makedrop_event,
                                        opening_date, public_date,
                                        closing_date, expected):
    client, session = client_session

    event = get_event(session, 'iris_test_4event')
    event.opening_timestamp = opening_date
    event.public_opening_timestamp = public_date
    event.closing_timestamp = closing_date
    session.commit()
    sign_up_team(session, 'iris_test_4event', 'test_user')

    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('http://localhost/events/iris_test_4event/sandbox')
        assert rv.status_code == 200
        # check for update button status on the generated .html
        if expected == b'event-close':
            assert 'disabled' in str(rv.data)  # should to be disabled
        else:
            assert 'disabled' not in str(rv.data)  # should not be disabled


@pytest.mark.parametrize(
    "submission_dir, filename",
    [("submissions/starting_kit", "classifier2.py"),
     ("/", "README12.md"),
     ("submissions/starting_kit", "clasifier.py")]
)
def test_sandbox_upload_file_dont_exist(client_session, makedrop_event,
                                        submission_dir, filename):
    client, session = client_session
    sign_up_team(session, "iris_test_4event", "test_user")

    config = ramp_config_template()
    ramp_config = generate_ramp_config(read_config(config))

    # upload file in sandbox.html
    path_submissions = os.path.join(ramp_config["ramp_kit_dir"],)

    with login_scope(client, "test_user", "test") as client:
        rv = client.get("http://localhost/events/iris_test_4event/sandbox")
        assert rv.status_code == 200

        # choose file and check if it was uploaded correctly
        path_submission = os.path.join(path_submissions, filename)
        assert not os.path.isfile(path_submission)

        with pytest.raises(FileNotFoundError):
            rv = client.post(
                "http://localhost/events/iris_test_4event/sandbox",
                headers={
                    "Referer":
                    "http://localhost/events/iris_test_4event/sandbox"
                },
                data={"file": (open(path_submission, "rb"), filename)},
                follow_redirects=False,
            )

        assert rv.status_code == 200

        with pytest.raises(FileNotFoundError):
            with open(path_submission, "r") as file:
                submitted_data = file.read()
                assert not submitted_data

        # get user interactions from db and check if 'upload' was added
        user_interactions = get_user_interactions_by_name(session, "test_user")
        assert "upload" not in user_interactions["interaction"].values


@pytest.mark.parametrize(
    "submission_dir, filename", [("/", "README.md"), ("/", "requirements.txt")]
)
def test_sandbox_upload_file_wrong(client_session, makedrop_event,
                                   submission_dir, filename):
    client, session = client_session
    sign_up_team(session, "iris_test_4event", "test_user")

    config = ramp_config_template()
    ramp_config = generate_ramp_config(read_config(config))

    # upload file in sandbox.html
    path_submissions = os.path.join(ramp_config["ramp_kit_dir"],)

    with login_scope(client, "test_user", "test") as client:
        rv = client.get("http://localhost/events/iris_test_4event/sandbox")
        assert rv.status_code == 200

        # choose file and check if it was uploaded correctly
        path_submission = os.path.join(path_submissions, filename)
        assert os.path.isfile(path_submission)

        rv = client.post(
            "http://localhost/events/iris_test_4event/sandbox",
            headers={"Referer":
                     "http://localhost/events/iris_test_4event/sandbox"},
            data={"file": (open(path_submission, "rb"), filename)},
            follow_redirects=False,
        )

        assert rv.status_code == 302
        assert (rv.location ==
                "http://localhost/events/iris_test_4event/sandbox")

        with open(path_submission, "r") as file:
            submitted_data = file.read()
        assert submitted_data

        # get user interactions from db and check if 'upload' was added
        user_interactions = get_user_interactions_by_name(session, "test_user")
        assert "upload" not in user_interactions["interaction"].values


@pytest.mark.parametrize(
    "submission_dir, filename",
    [("submissions/error", "estimator.py"),
     ("submissions/random_forest_10_10", "estimator.py"),
     ("submissions/starting_kit", "estimator.py")]
)
def test_sandbox_upload_file(client_session, makedrop_event,
                             submission_dir, filename):
    client, session = client_session
    sign_up_team(session, "iris_test_4event", "test_user")

    config = ramp_config_template()
    ramp_config = generate_ramp_config(read_config(config))

    # upload file in sandbox.html
    path_submissions = os.path.join(ramp_config["ramp_kit_dir"],
                                    submission_dir)

    with login_scope(client, "test_user", "test") as client:
        rv = client.get("http://localhost/events/iris_test_4event/sandbox")
        assert rv.status_code == 200

        # choose file and check if it was uploaded correctly
        path_submission = os.path.join(path_submissions, filename)
        assert os.path.isfile(path_submission)

        rv = client.post(
            "http://localhost/events/iris_test_4event/sandbox",
            headers={
                     "Referer":
                     "http://localhost/events/iris_test_4event/sandbox"},
            data={"file": (open(path_submission, "rb"), filename)},
            follow_redirects=False,
        )

        assert rv.status_code == 302
        assert (rv.location ==
                "http://localhost/events/iris_test_4event/sandbox")

        # code of the saved file
        with open(path_submission, "r") as file:
            submitted_data = file.read()

        # code from the db
        event = get_event(session, "iris_test_4event")
        sandbox_submission = get_submission_by_name(
            session, "iris_test_4event", "test_user", event.ramp_sandbox_name
        )
        submission_code = sandbox_submission.files[-1].get_code()

        # get user interactions from db and check if 'upload' was added
        user_interactions = get_user_interactions_by_name(session, "test_user")

        # check if the code of the submitted file in the 'submission_code'
        assert submitted_data is not None
        assert submitted_data in submission_code
        # check if the user_interaction was added to the db
        assert "upload" in user_interactions["interaction"].values


def test_sandbox_save_file(client_session, makedrop_event):
    client, session = client_session
    sign_up_team(session, "iris_test_4event", "test_user")

    example_code = "example content"

    with login_scope(client, "test_user", "test") as client:
        rv = client.get("http://localhost/events/iris_test_4event/sandbox")
        assert rv.status_code == 200

        rv = client.post(
            "http://localhost/events/iris_test_4event/sandbox",
            headers={"Referer":
                     "http://localhost/events/iris_test_4event/sandbox"},
            data={"estimator": example_code,
                  "code-csrf_token": "temp_token"},
            follow_redirects=False,
        )
        assert rv.status_code == 200

        # code from the db
        event = get_event(session, "iris_test_4event")
        sandbox_submission = get_submission_by_name(
            session, "iris_test_4event", "test_user", event.ramp_sandbox_name
        )
        submission_code = sandbox_submission.files[-1].get_code()

        # get user interactions from db and check if 'save' was added
        user_interactions = get_user_interactions_by_name(session, "test_user")

        assert "save" in user_interactions["interaction"].values
        assert example_code in submission_code

    # make sure that after changing the code example
    # and reloading the page the code is still changed
    with login_scope(client, "test_user", "test") as client:
        rv = client.get("http://localhost/events/iris_test_4event/sandbox")
        assert rv.status_code == 200
        assert example_code.encode() in rv.data


@pytest.mark.parametrize(
    "opening_date, public_date, closing_date, expected", testtimestamps
)
def test_correct_message_sandbox(client_session, makedrop_event,
                                 opening_date, public_date,
                                 closing_date, expected):
    client, session = client_session

    event = get_event(session, 'iris_test_4event')
    event.opening_timestamp = opening_date
    event.public_opening_timestamp = public_date
    event.closing_timestamp = closing_date
    session.commit()
    sign_up_team(session, 'iris_test_4event', 'test_user')

    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('http://localhost/events/iris_test_4event/sandbox')
        assert rv.status_code == 200

        if NOW < opening_date:
            assert "Event submissions will open on the " in str(rv.data)
        elif NOW < closing_date:
            assert "Event submissions are open until " in str(rv.data)
        else:
            assert "This event closed on the " in str(rv.data)


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
            rv = client.get('{}/{}'.format(submission_hash, 'estimator.py'))
            assert rv.status_code == 302
            assert rv.location == 'http://localhost/problems'
            with client.session_transaction() as cs:
                flash_message = dict(cs['_flashes'])
            assert "does not exist by" in flash_message['message']
    finally:
        os.rename(submission.path + 'xxxxx', submission.path)

    # GET: normal file display
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('{}/{}'.format(submission_hash, 'estimator.py'))
        assert rv.status_code == 200
        assert b'file = estimator.py' in rv.data
        assert (b'from sklearn.ensemble import RandomForestClassifier' in
                rv.data)


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


def test_toggle_competition(client_session):
    client, session = client_session

    event = (session.query(Event)
                    .filter_by(name="iris_test")
                    .first())
    event.is_competitive = True
    session.commit()

    # unknown submission
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get("toggle_competition/xxxxx")
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "Missing submission" in flash_message['message']

    tmp_timestamp = event.closing_timestamp
    event.closing_timestamp = datetime.datetime.utcnow()
    session.commit()

    submission = (session.query(Submission)
                         .filter_by(name="starting_kit_test",
                                    event_team_id=1)
                         .first())

    # submission not accessible by the test user
    with login_scope(client, 'test_user_2', 'test') as client:
        rv = client.get("toggle_competition/{}".format(submission.hash_))
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "Missing submission" in flash_message['message']

    event.closing_timestamp = tmp_timestamp
    session.commit()

    # submission accessible by the user and check if we can add/remove from
    # competition
    with login_scope(client, 'test_user', 'test') as client:
        # check that the submission is tagged to be in the competition
        assert submission.is_in_competition
        rv = client.get('{}/{}'.format(submission.hash_, 'estimator.py'))
        assert b"Pull out this submission from the competition" in rv.data
        # trigger the pull-out of the competition
        rv = client.get("toggle_competition/{}".format(submission.hash_))
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/{}/estimator.py'.format(
            submission.hash_)
        rv = client.get(rv.location)
        assert b"Enter this submission into the competition" in rv.data
        session.commit()
        assert not submission.is_in_competition
        # trigger the entering in the competition
        rv = client.get("toggle_competition/{}".format(submission.hash_))
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/{}/estimator.py'.format(
            submission.hash_)
        rv = client.get(rv.location)
        assert b"Pull out this submission from the competition" in rv.data
        session.commit()
        assert submission.is_in_competition


def test_download_submission(client_session):
    client, session = client_session

    # unknown submission
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get("download/xxxxx")
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert "Missing submission" in flash_message['message']

    submission = (session.query(Submission)
                         .filter_by(name="starting_kit_test",
                                    event_team_id=1)
                         .first())

    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get(f"download/{submission.hash_}")
        assert rv.status_code == 200
        assert rv.data
