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


def test_login(client_session):
    client, session = client_session

    # GET without any previous login
    rv = client.get('/login')
    assert rv.status_code == 200
    assert b'Login' in rv.data
    assert b'Username' in rv.data
    assert b'Password' in rv.data

    # GET with a previous login
    login(client, 'test_user', 'test')
    rv = client.get('/login')
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/problems'
    rv = client.get('/login', follow_redirects=True)
    assert rv.status_code == 200
    logout(client)

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
    login(client, 'test_user', 'test')
    rv = getattr(client, request_function)('/sign_up')
    assert rv.status_code == 302
    assert rv.location == 'http://localhost/problems'
    rv = getattr(client, request_function)('/sign_up', follow_redirects=True)
    assert rv.status_code == 200
    logout(client)


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

    login(client, 'test_user', 'test')
    # GET function once logged-in
    rv = client.get('/update_profile')
    assert rv.status_code == 200
    for attr in [b'Username', b'First name', b'Last name', b'Email', b'User',
                 b'Test', b'test.user@gmail.com']:
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
