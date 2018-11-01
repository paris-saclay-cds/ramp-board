from __future__ import print_function, unicode_literals

import logging
import os
import sys
from distutils.util import strtobool

from sqlalchemy.exc import IntegrityError

from termcolor import colored

logger = logging.getLogger('databoard')


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

    if port is None:
        port = app.config.get('RAMP_SERVER_PORT')
    server_port = int(port)
    app.run(
        debug=False,
        port=server_port,
        use_reloader=False,
        host='0.0.0.0',
        processes=1000,
        threaded=False)


def profile(port=None, profiling_file='profiler.log'):
    from werkzeug.contrib.profiler import ProfilerMiddleware
    from werkzeug.contrib.profiler import MergeStream
    from databoard import app

    app.config['PROFILE'] = True
    f = open(profiling_file, 'w')
    stream = MergeStream(sys.stdout, f)
    app.wsgi_app = ProfilerMiddleware(app.wsgi_app, stream=stream,
                                      restrictions=[30])
    if port is None:
        port = app.config.get('RAMP_SERVER_PORT')
        server_port = int(port)
        app.run(debug=True,
                port=server_port,
                use_reloader=False,
                host='0.0.0.0',
                processes=1000)


def add_problem(name, force='False'):
    """Add new problem.

    If force=True, deletes problem (with all events) if exists.
    """
    force = strtobool(force)

    from databoard.db_tools import add_problem

    add_problem(name, force)


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


def train_test(e, t=None, s=None, state=None, force='False',
               is_save_y_pred='False', is_parallelize=''):
    force = strtobool(force)
    if is_parallelize == '':
        is_parallelize = None
    else:
        is_parallelize = strtobool(is_parallelize)

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
    train_test_submissions(submissions, force_retrain_test=force,
                           is_parallelize=is_parallelize)
    compute_contributivity(e, is_save_y_pred=is_save_y_pred)


def score_submission(e, t, s, is_save_y_pred='False'):
    from databoard.db_tools import score_submission, get_submissions

    submissions = get_submissions(
        event_name=e, team_name=t, submission_name=s)
    score_submission(submissions[0])
    compute_contributivity(e, is_save_y_pred=is_save_y_pred)


def set_n_submissions(e=None):
    from databoard.db_tools import set_n_submissions
    set_n_submissions(e)


def compute_contributivity(e, is_save_y_pred='False'):
    is_save_y_pred = strtobool(is_save_y_pred)

    from databoard.db_tools import compute_contributivity
    from databoard.db_tools import compute_historical_contributivity
    compute_contributivity(e, is_save_y_pred=is_save_y_pred)
    compute_historical_contributivity(e)
    set_n_submissions(e)


def delete_submission(e, t, s):
    from databoard.db_tools import delete_submission
    delete_submission(event_name=e, team_name=t, submission_name=s)


def create_user(name, password, lastname, firstname, email,
                access_level='user', hidden_notes=''):
    from databoard.db_tools import create_user
    user = create_user(
        name, password, lastname, firstname, email, access_level, hidden_notes)
    return user


def generate_single_password():
    from databoard.utils import generate_single_password
    print(generate_single_password())


def generate_passwords(users_to_add_f_name, password_f_name):
    from databoard.utils import generate_passwords
    print(generate_passwords(users_to_add_f_name, password_f_name))


def add_users_from_file(users_to_add_f_name, password_f_name):
    """Add users.

    Users whould be the same in the same order in the two files.
    """
    import pandas as pd
    from databoard.model import NameClashError
    from databoard.utils import remove_non_ascii

    users_to_add = pd.read_csv(users_to_add_f_name)
    passwords = pd.read_csv(password_f_name)
    users_to_add['password'] = passwords['password']
    ids = []
    for _, u in users_to_add.iterrows():
        print(u)
        try:
            if 'access_level' in u:
                acces_level = u['access_level']
            else:
                acces_level = 'user'
            user = create_user(
                remove_non_ascii(u['name']), u['password'],
                u['lastname'], u['firstname'], u['email'], acces_level,
                u['hidden_notes'])
            ids.append(user.id)
        except NameClashError:
            print(colored(
                'user {}:{} already in database'.format(u.name, u.email),
                'red'))
    for id in ids:
        print(id)


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
    import pandas as pd
    from databoard.model import DuplicateSubmissionError
    from databoard.utils import remove_non_ascii

    users_to_sign_up = pd.read_csv(users_to_add_f_name)
    for _, u in users_to_sign_up.iterrows():
        username = remove_non_ascii(u['name'])
        print('signing up {} to {}'.format(username.encode('utf-8'), event))
        try:
            sign_up_team(event, username)
        except DuplicateSubmissionError:
            print(colored(
                'user {}:{} already signed up'.format(username, u.email),
                'red'))


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


def backend_train_test_loop(e=None, timeout=30,
                            is_compute_contributivity='True',
                            is_parallelize=''):
    """Automated training loop.

    Picks up the earliest new submission and trains it, in an infinite
    loop.

    Parameters
    ----------
    e : string
        Event name. If set, only train submissions from that event.
        If event name is prefixed by not, it excludes that event.
    """
    if is_parallelize == '':
        is_parallelize = None
    else:
        is_parallelize = strtobool(is_parallelize)

    from databoard.db_tools import backend_train_test_loop
    is_compute_contributivity = strtobool(is_compute_contributivity)
    backend_train_test_loop(
        e, timeout, is_compute_contributivity, is_parallelize)


def set_state(e, t, s, state):
    from databoard.db_tools import set_state
    set_state(e, t, s, state)


def exclude_from_ensemble(e, t, s):
    from databoard.db_tools import exclude_from_ensemble
    exclude_from_ensemble(e, t, s)
