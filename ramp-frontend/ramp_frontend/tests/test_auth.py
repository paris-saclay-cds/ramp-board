import re
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
from ramp_database.utils import check_password

from ramp_database.tools.user import get_user_by_name
from ramp_database.tools.user import get_user_by_name_or_email

from ramp_frontend import create_app
from ramp_frontend.testing import login_scope
from ramp_frontend.testing import login
from ramp_frontend.testing import logout
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


def test_login(client_session):
    client, session = client_session

    # GET without any previous login
    rv = client.get('/login')
    assert rv.status_code == 200
    assert b'Login' in rv.data
    assert b'Username' in rv.data
    assert b'Password' in rv.data

    # GET with a previous login
    with login_scope(client, 'test_user', 'test') as client:
        rv = client.get('/login')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        rv = client.get('/login', follow_redirects=True)
        assert rv.status_code == 200

    # POST with unknown username
    login_info = {'user_name': 'unknown', 'password': 'xxx'}
    rv = client.post('/login', data=login_info)
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert flash_message['message'] == 'User "unknown" does not exist'
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/login'
    rv = client.post('/login', data=login_info, follow_redirects=True)
    assert rv.status_code == 200

    # POST with wrong password
    login_info = {'user_name': 'test_user', 'password': 'xxx'}
    rv = client.post('/login', data=login_info)
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert flash_message['message'] == 'Wrong password'
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/login'
    rv = client.post('/login', data=login_info, follow_redirects=True)
    assert rv.status_code == 200

    # POST with a right login and password
    login_info = {'user_name': 'test_user', 'password': 'test'}
    rv = client.post('/login', data=login_info)
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/problems'
    user = get_user_by_name_or_email(session, login_info['user_name'])
    assert user.is_authenticated
    logout(client)
    rv = client.post('/login', data=login_info, follow_redirects=True)
    assert rv.status_code == 200
    logout(client)

    # POST with a right email as login and password
    login_info = {'user_name': 'test_user', 'password': 'test',
                  'email': 'test.user@gmail.com'}
    rv = client.post('/login', data=login_info)
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/problems'
    user = get_user_by_name_or_email(session,
                                     login_info['email'])
    assert user.is_authenticated
    logout(client)
    rv = client.post('/login', data=login_info, follow_redirects=True)
    assert rv.status_code == 200
    logout(client)

    # POST with right login and password from a different location webpage
    login_info = {'user_name': 'test_user', 'password': 'test'}
    landing_page = {'next': 'http://localhost/events/iris_test'}
    rv = client.post('/login', data=login_info, query_string=landing_page)
    assert rv.status_code == 302
    assert rv.location == landing_page['next']
    logout(client)
    rv = client.post('/login', data=login_info, query_string=landing_page,
                     follow_redirects=True)
    assert rv.status_code == 200
    logout(client)


def test_logout(client_session):
    client, session = client_session

    # logout without previous login
    rv = client.get('/logout')
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/login?next=%2Flogout'
    rv = client.get('/logout', follow_redirects=True)
    assert rv.status_code == 200

    # logout from a previous login
    login(client, 'test_user', 'test')
    rv = client.get('/logout')
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/login'
    user = get_user_by_name(session, 'test_user')
    assert not user.is_authenticated
    login(client, 'test_user', 'test')
    rv = client.get('/logout', follow_redirects=True)
    assert rv.status_code == 200


@pytest.mark.parametrize("request_function", ['get', 'post'])
def test_sign_up_already_logged_in(client_session, request_function):
    client, _ = client_session

    # sign-up when already logged-in
    with login_scope(client, 'test_user', 'test') as client:
        rv = getattr(client, request_function)('/sign_up')
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        rv = getattr(client, request_function)('/sign_up',
                                               follow_redirects=True)
        assert rv.status_code == 200


def test_sign_up(client_session):
    client, session = client_session

    # GET on sign-up
    rv = client.get('/sign_up')
    assert rv.status_code == 200
    assert b'Sign Up' in rv.data

    # POST on sign-up
    user_profile = {'user_name': 'xx', 'password': 'xx', 'firstname': 'xx',
                    'lastname': 'xx', 'email': 'xx'}
    rv = client.post('/sign_up', data=user_profile)
    assert rv.status_code == 302
    user = get_user_by_name(session, 'xx')
    assert user.name == 'xx'
    user_profile = {'user_name': 'yy', 'password': 'yy', 'firstname': 'yy',
                    'lastname': 'yy', 'email': 'yy'}
    rv = client.post('/sign_up', data=user_profile, follow_redirects=True)
    assert rv.status_code == 200

    def _assert_flash(url, data, status_code=302,
                      message='username is already in use'):
        rv = client.post('/sign_up', data=data)
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert (flash_message['message'] == message)
        assert rv.status_code == status_code

    # check that we catch a flash error if we try to sign-up with an identical
    # username
    user_profile = {'user_name': 'xx', 'password': 'xx',
                    'firstname': 'xx', 'lastname': 'xx',
                    'email': 'test_user@gmail.com'}
    _assert_flash('/sign_up', data=user_profile,
                  message='username is already in use')

    user_profile.update(user_name='new', email="yy")
    _assert_flash('/sign_up', data=user_profile,
                  message='email is already in use')

    user_profile.update(user_name='yy', email="yy")
    _assert_flash('/sign_up', data=user_profile,
                  message=("username is already in use "
                           "and email is already in use"))


@_fail_no_smtp_server
def test_sign_up_with_approval(client_session):
    # check the sign-up and email confirmation framework
    client, session = client_session

    with client.application.app_context():
        with mail.record_messages() as outbox:
            user_profile = {
                'user_name': 'new_user_1', 'password': 'xx', 'firstname': 'xx',
                'lastname': 'xx', 'email': 'new_user_1@mail.com'
            }
            rv = client.post('/sign_up', data=user_profile)
            # check the flash box to inform the user about the mail
            with client.session_transaction() as cs:
                flash_message = dict(cs['_flashes'])
            assert 'We sent a confirmation email.' in flash_message['message']
            # check that the email has been sent
            assert len(outbox) == 1
            assert ('Click on the following link to confirm your email'
                    in outbox[0].body)
            # get the link to reset the password
            reg_exp = re.search(
                "http://localhost/confirm_email/.*", outbox[0].body
            )
            confirm_email_link = reg_exp.group()
            # remove the part with 'localhost' for the next query
            confirm_email_link = confirm_email_link[
                confirm_email_link.find('/confirm_email'):
            ]
            # check the redirection
            assert rv.status_code == 302
            user = get_user_by_name(session, 'new_user_1')
            assert user is not None
            assert user.access_level == 'not_confirmed'

    # POST method of the email confirmation
    with client.application.app_context():
        with mail.record_messages() as outbox:
            rv = client.post(confirm_email_link)
            # check the flash box to inform the user to wait for admin's
            # approval
            with client.session_transaction() as cs:
                flash_message = dict(cs['_flashes'])
            assert ('An email has been sent to the RAMP administrator' in
                    flash_message['message'])
            # check that we send an email to the administrator
            assert len(outbox) == 1
            assert "Approve registration of new_user_1" in outbox[0].subject
            # ensure that we have the last changes
            session.commit()
            user = get_user_by_name(session, 'new_user_1')
            assert user.access_level == 'asked'
            assert rv.status_code == 302
            assert rv.location == 'http://localhost/login'

    # POST to check that we raise the right errors
    # resend the confirmation for a user which already confirmed
    rv = client.post(confirm_email_link)
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert ('Your email address already has been confirmed'
            in flash_message['error'])
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/'
    # check when the user was already approved
    for status in ('user', 'admin'):
        user = get_user_by_name(session, 'new_user_1')
        user.access_level = status
        session.commit()
        rv = client.post(confirm_email_link)
        with client.session_transaction() as cs:
            flash_message = dict(cs['_flashes'])
        assert 'Your account is already approved.' in flash_message['error']
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/login'
    # delete the user in the middle
    session.delete(user)
    session.commit()
    rv = client.post(confirm_email_link)
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert 'You did not sign-up yet to RAMP.' in flash_message['error']
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/sign_up'
    # access a token which does not exist
    rv = client.post('/confirm_email/xxx')
    assert rv.status_code == 404


def test_update_profile(client_session):
    client, session = client_session

    # try to change the profile without being logged-in
    rv = client.get('/update_profile')
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/login?next=%2Fupdate_profile'
    rv = client.get('/update_profile', follow_redirects=True)
    assert rv.status_code == 200

    with login_scope(client, 'test_user', 'test') as client:
        # GET function once logged-in
        rv = client.get('/update_profile')
        assert rv.status_code == 200
        for attr in [b'Username', b'First name', b'Last name', b'Email',
                     b'User', b'Test', b'test.user@gmail.com']:
            assert attr in rv.data

        # POST function once logged-in
        user_profile = {'lastname': 'XXX', 'firstname': 'YYY',
                        'email': 'xxx@gmail.com'}
        rv = client.post('/update_profile', data=user_profile)
        assert rv.status_code == 302
        assert rv.location == 'http://localhost/problems'
        user = get_user_by_name(session, 'test_user')
        assert user.lastname == 'XXX'
        assert user.firstname == 'YYY'
        assert user.email == 'xxx@gmail.com'
        user_profile = {'lastname': 'Test', 'firstname': 'User',
                        'email': 'test.user@gmail.com'}
        rv = client.post('/update_profile', data=user_profile,
                         follow_redirects=True)
        assert rv.status_code == 200


@_fail_no_smtp_server
def test_reset_password(client_session):
    client, session = client_session

    # GET method
    rv = client.get('/reset_password')
    assert rv.status_code == 200
    assert b'If you are a registered user, we are going to send' in rv.data

    # POST method
    # check that we raise an error if the email does not exist
    rv = client.post('/reset_password', data={'email': 'random@mail.com'})
    assert rv.status_code == 200
    assert b'You can sign-up instead.' in rv.data

    # set a user to "asked" access level
    user = get_user_by_name(session, 'test_user')
    user.access_level = 'asked'
    session.commit()
    rv = client.post('/reset_password', data={'email': user.email})
    assert rv.status_code == 200
    assert b'Your account has not been yet approved.' in rv.data

    # set back the account to 'user' access level
    user.access_level = 'user'
    session.commit()
    rv = client.post('/reset_password', data={'email': user.email})
    with client.session_transaction() as cs:
        flash_message = dict(cs['_flashes'])
    assert flash_message['message'] == ('An email to reset your password has '
                                        'been sent')
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/login'

    with client.application.app_context():
        with mail.record_messages() as outbox:
            rv = client.post('/reset_password', data={'email': user.email})
            assert len(outbox) == 1
            assert 'click on the link to reset your password' in outbox[0].body
            # get the link to reset the password
            reg_exp = re.search(
                "http://localhost/reset/.*", outbox[0].body
            )
            reset_password_link = reg_exp.group()
            # remove the part with 'localhost' for the next query
            reset_password_link = reset_password_link[
                reset_password_link.find('/reset'):
            ]

    # check that we can reset the password using the previous link
    # GET method
    rv = client.get(reset_password_link)
    assert rv.status_code == 200
    assert b'Change my password' in rv.data

    # POST method
    new_password = 'new_password'
    rv = client.post(reset_password_link, data={'password': new_password})
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/login'
    # make a commit to be sure that the update has been done
    session.commit()
    user = get_user_by_name(session, 'test_user')
    assert check_password(new_password, user.hashed_password)


@_fail_no_smtp_server
def test_reset_token_error(client_session):
    client, session = client_session

    # POST method
    new_password = 'new_password'
    rv = client.post('/reset/xxx', data={'password': new_password})
    assert rv.status_code == 404

    # Get get the link to a real token but remove the user in between
    user = get_user_by_name(session, 'test_user')
    with client.application.app_context():
        with mail.record_messages() as outbox:
            rv = client.post('/reset_password', data={'email': user.email})
            assert len(outbox) == 1
            assert 'click on the link to reset your password' in outbox[0].body
            # get the link to reset the password
            reg_exp = re.search(
                "http://localhost/reset/.*", outbox[0].body
            )
            reset_password_link = reg_exp.group()
            # remove the part with 'localhost' for the next query
            reset_password_link = reset_password_link[
                reset_password_link.find('/reset'):
            ]

    user = get_user_by_name(session, 'test_user')
    session.delete(user)
    session.commit()
    new_password = 'new_password'
    rv = client.post(reset_password_link, data={'password': new_password})
    assert rv.status_code == 404
