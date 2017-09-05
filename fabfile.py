import sys
import pandas as pd
import os
import logging
from distutils.util import strtobool
from sqlalchemy.exc import IntegrityError

logger = logging.getLogger('databoard')


def add_test_users():
    import databoard.db_tools as db_tools

    db_tools.create_user(
        name='kegl', password='pwd',
        lastname='Kegl', firstname='Balazs',
        email='balazs.kegl@gmail.com', access_level='admin')
    db_tools.create_user(
        name='agramfort', password='pwd',
        lastname='Gramfort', firstname='Alexandre',
        email='alexandre.gramfort@gmail.com', access_level='admin')
    db_tools.create_user(
        name='akazakci', password='pwd',
        lastname='Akin', firstname='Kazakci',
        email='osmanakin@gmail.com', access_level='admin')
    db_tools.create_user(
        name='mcherti', password='pwd', lastname='Cherti',
        firstname='Mehdi', email='mehdicherti@gmail.com',
        access_level='admin')
    db_tools.create_user(
        name='test_user', password='test',
        lastname='Test', firstname='User',
        email='test.user@gmail.com', access_level='user')
    db_tools.create_user(
        name='test_iris_admin', password='test',
        lastname='Admin', firstname='Iris',
        email='iris.admin@gmail.com', access_level='user')


def deploy_locally():
    import databoard.config as config
    from databoard.remove_test_db import recreate_test_db
    import databoard.db_tools as db_tools

    os.system('rm -rf ' + config.local_test_deployment_path)
    os.makedirs(config.local_test_deployment_path)
    os.system('rsync -rultv fabfile.py ' + config.deployment_path)
    os.makedirs(config.ramp_kits_path)
    os.makedirs(config.ramp_data_path)
    os.makedirs(config.submissions_path)

    recreate_test_db()
    add_test_users()
    db_tools.setup_workflows()


def test_keywords():
    import databoard.db_tools as db_tools
    db_tools.add_keyword('botany', 'data_domain', 'scientific data', 'Botany.')
    db_tools.add_keyword(
        'real estate', 'data_domain', 'industrial data', 'Real estate.')
    db_tools.add_keyword(
        'regression', 'data_science_theme', None, 'Regression.')
    db_tools.add_keyword(
        'classification', 'data_science_theme', None, 'Classification.')
    db_tools.add_problem_keyword('iris', 'classification')
    db_tools.add_problem_keyword('iris', 'botany')
    db_tools.add_problem_keyword('boston_housing', 'regression')
    db_tools.add_problem_keyword('boston_housing', 'real estate')


def test_make_event_admin():
    import databoard.db_tools as db_tools
    db_tools.make_event_admin('iris', 'test_iris_admin')


def test_problem(problem_name, test_user_name):
    add_problem(problem_name, with_download='True')
    event_name = '{}'.format(problem_name)
    event_title = 'test event'
    add_event(problem_name, event_name, event_title, is_public="True")
    sign_up_team(event_name, test_user_name)
    submit_starting_kit(event_name, test_user_name)
    train_test(event_name, test_user_name)
    update_leaderboards(event_name)
    update_user_leaderboards(event_name, test_user_name)


def submit_starting_kit(e, t):
    from databoard.db_tools import submit_starting_kit
    submit_starting_kit(event_name=e, team_name=t)


def sign_up_team(e, t):
    from databoard.db_tools import sign_up_team, get_submissions
    sign_up_team(event_name=e, team_name=t)
    if not os.environ.get('DATABOARD_TEST'):
        submission = get_submissions(event_name=e, team_name=t,
                                     submission_name="starting_kit")[0]
        os.system('sudo chown -R www-data:www-data %s' % submission.path)


def approve_user(u):
    from databoard.db_tools import approve_user
    approve_user(user_name=u)


def serve(port=None):
    from databoard import app
    import databoard.views  # noqa
    import databoard.config as config

    if port is None:
        port = config.server_port
    server_port = int(port)
    app.run(
        debug=False,
        port=server_port,
        use_reloader=False,
        host='0.0.0.0',
        processes=1000)


def profile(port=None, profiling_file='profiler.log'):
    from werkzeug.contrib.profiler import ProfilerMiddleware
    from werkzeug.contrib.profiler import MergeStream
    from databoard import app
    import databoard.config as config

    app.config['PROFILE'] = True
    f = open(profiling_file, 'w')
    stream = MergeStream(sys.stdout, f)
    app.wsgi_app = ProfilerMiddleware(app.wsgi_app, stream=stream,
                                      restrictions=[30])
    if port is None:
        port = config.server_port
        server_port = int(port)
        app.run(debug=True,
                port=server_port,
                use_reloader=False,
                host='0.0.0.0',
                processes=1000)


def add_score_type(name, is_lower_the_better, minimum, maximum):
    from databoard.db_tools import add_score_type

    add_score_type(
        name, is_lower_the_better, float(minimum), float(maximum))


def add_problem(name, force='False', with_download='False'):
    """Add new problem.

    If force=True, deletes problem (with all events) if exists.
    """
    force = strtobool(force)
    with_download = strtobool(with_download)

    from databoard.db_tools import add_problem

    add_problem(name, force, with_download)


def add_event(problem_name, event_name, event_title, is_public='True',
              force='False'):
    """Add new event.

    If force=True, deletes event (with all submissions) if exists.
    """
    force = bool(strtobool(force))
    is_public = bool(strtobool(is_public))
    from databoard.db_tools import add_event
    try:
        add_event(problem_name, event_name, event_title, is_public, force)
    except IntegrityError:
        logger.info(
            'Attempting to delete event, use "force=True" ' +
            'if you know what you are doing')


def make_event_admin(e, u):
    from databoard.db_tools import make_event_admin

    make_event_admin(event_name=e, admin_name=u)


def train_test(e, t=None, s=None, state=None, force='False'):
    force = strtobool(force)

    from databoard.db_tools import train_test_submissions,\
        get_submissions, get_submissions_of_state
    from databoard.config import sandbox_d_name

    if state is not None:
        submissions = get_submissions_of_state(state)
    else:
        submissions = get_submissions(
            event_name=e, team_name=t, submission_name=s)
    submissions = [submission for submission in submissions
                   if submission.name != sandbox_d_name]
    train_test_submissions(submissions, force_retrain_test=force)
    compute_contributivity(e)


def set_n_submissions(e=None):
    from databoard.db_tools import set_n_submissions
    set_n_submissions(e)


def compute_contributivity(e):
    from databoard.db_tools import compute_contributivity
    from databoard.db_tools import compute_historical_contributivity
    compute_contributivity(e)
    compute_historical_contributivity(e)
    set_n_submissions(e)


def print_submissions(e=None, t=None, s=None):
    from databoard.db_tools import print_submissions
    print_submissions(event_name=e, team_name=t, submission_name=s)


def print_submission_similaritys():
    from databoard.db_tools import print_submission_similaritys
    print_submission_similaritys()


def delete_submission(e, t, s):
    from databoard.db_tools import delete_submission
    delete_submission(event_name=e, team_name=t, submission_name=s)


def create_user(name, password, lastname, firstname, email,
                access_level='user', hidden_notes=''):
    from databoard.db_tools import create_user
    create_user(
        name, password, lastname, firstname, email, access_level, hidden_notes)


def generate_single_password():
    from databoard.db_tools import generate_single_password
    print(generate_single_password())


def generate_passwords(users_to_add_f_name, password_f_name):
    from databoard.db_tools import generate_passwords
    print(generate_passwords(users_to_add_f_name, password_f_name))


def add_users_from_file(users_to_add_f_name, password_f_name):
    """Add users.

    Users whould be the same in the same order in the two files.
    """
    from databoard.model import NameClashError

    users_to_add = pd.read_csv(users_to_add_f_name)
    passwords = pd.read_csv(password_f_name)
    users_to_add['password'] = passwords['password']
    for _, u in users_to_add.iterrows():
        print u
        try:
            if 'access_level' in u:
                acces_level = u['access_level']
            else:
                acces_level = 'user'
            create_user(
                u['name'], u['password'], u['lastname'], u['firstname'],
                u['email'], acces_level, u['hidden_notes'])
        except NameClashError:
            print 'user already in database'


def send_password_mail(user_name, password):
    """Update <user_name>'s password to <password> and mail it to him/her.

    Parameters
    ----------
    user_name : user name
    password : new password
    """
    from databoard.db_tools import send_password_mail
    send_password_mail(user_name, password)


def send_password_mails(password_f_name):
    """Update <name>'s password to <password>, read from <password_f_name>.

    Can be generated by `generate_passwords <generate_passwords>`.
    Parameters
    ----------
    password_f_name : a csv file with columns `name` and `password`
    """
    from databoard.db_tools import send_password_mails
    send_password_mails(password_f_name)


def sign_up_event_users_from_file(users_to_add_f_name, event):
    from databoard.model import DuplicateSubmissionError
    users_to_sign_up = pd.read_csv(users_to_add_f_name)
    for _, u in users_to_sign_up.iterrows():
        print 'signing up {} to {}'.format(u['name'], event)
        try:
            sign_up_team(event, u['name'])
        except DuplicateSubmissionError:
            print 'user already signed up'


def dump_user_interactions():
    from databoard.db_tools import get_user_interactions_df
    user_interactions_df = get_user_interactions_df()
    user_interactions_df.to_csv('user_interactions_dump.csv')


def update_leaderboards(e=None):
    from databoard.model import Event
    from databoard.db_tools import update_leaderboards
    if e is None:
        es = Event.query.all()
        for e in es:
            update_leaderboards(e.name)
    else:
        update_leaderboards(e)


def update_user_leaderboards(e, u):
    from databoard.db_tools import update_user_leaderboards
    update_user_leaderboards(e, u)


def update_all_user_leaderboards(e=None):
    from databoard.model import Event
    from databoard.db_tools import update_all_user_leaderboards
    if e is None:
        es = Event.query.all()
        for e in es:
            update_all_user_leaderboards(e.name)
    else:
        update_all_user_leaderboards(e)


def prepare_data(problem_name):
    from databoard.model import Problem
    problem = Problem.query.filter_by(name=problem_name).one_or_none()
    problem.module.prepare_data()


def backend_train_test_loop(e=None, timeout=30,
                            is_compute_contributivity='True'):
    """Automated training loop.

    Picks up the earliest new submission and trains it, in an infinite
    loop.

    Parameters
    ----------
    e : string
        Event name. If set, only train submissions from that event.
        If event name is prefixed by not, it excludes that event.
    """
    from databoard.db_tools import backend_train_test_loop
    is_compute_contributivity = strtobool(is_compute_contributivity)
    backend_train_test_loop(e, timeout, is_compute_contributivity)


def set_state(e, t, s, state):
    from databoard.db_tools import set_state
    set_state(e, t, s, state)

# The following function was implemented to handle user interaction dump
# but it turned out that the db insertion was not the CPU sink. Keep it
# for a while if the site is still slow.
# def update_user_interactions():
#     from databoard.db_tools import update_user_interactions
#     update_user_interactions()
