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
from ramp_frontend.testing import login_scope
from ramp_frontend.testing import login
from ramp_frontend.testing import logout


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
    user = get_user_by_name(session, login_info['user_name'])
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
        rv = getattr(client, request_function)('/sign_up', follow_redirects=True)
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
    assert rv.location == 'http://localhost/login'
    user = get_user_by_name(session, 'xx')
    assert user.name == 'xx'
    user_profile = {'user_name': 'yy', 'password': 'yy', 'firstname': 'yy',
                    'lastname': 'yy', 'email': 'yy'}
    rv = client.post('/sign_up', data=user_profile, follow_redirects=True)
    assert rv.status_code == 200


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
