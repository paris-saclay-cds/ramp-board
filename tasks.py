from __future__ import print_function, unicode_literals

import logging
import os
import sys

from invoke import task

logger = logging.getLogger('databoard')


@task
def sign_up_team(c, event, team):
    from databoard.db_tools import sign_up_team, get_submissions
    sign_up_team(event_name=event, team_name=team)
    if not os.environ.get('DATABOARD_TEST'):
        submissions = get_submissions(
            event_name=event, team_name=team, submission_name="starting_kit")
        c.run('sudo chown -R www-data:www-data %s' % submissions[0].path)


@task
def approve_user(c, user):
    from databoard.db_tools import approve_user
    approve_user(user_name=user)


@task
def serve(c, port=None):
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


@task
def profile(c, port=None, profiling_file='profiler.log'):
    from werkzeug.contrib.profiler import ProfilerMiddleware
    from werkzeug.contrib.profiler import MergeStream
    from databoard import app

    app.config['PROFILE'] = True
    with open(profiling_file, 'w') as f:
        stream = MergeStream(sys.stdout, f)
        app.wsgi_app = ProfilerMiddleware(
            app.wsgi_app, stream=stream, restrictions=[30])
        if port is None:
            port = app.config.get('RAMP_SERVER_PORT')
        server_port = int(port)
        app.run(
            debug=True,
            port=server_port,
            use_reloader=False,
            host='0.0.0.0',
            processes=1000)


@task
def add_problem(c, name, force=False):
    """Add new problem.

    If force=True, deletes problem (with all events) if exists.
    """
    from databoard.db_tools import add_problem

    add_problem(name, force)


@task
def add_event(c, problem_name, event_name, event_title, is_public=True,
              force=False):
    """Add new event.

    If force=True, deletes event (with all submissions) if exists.
    """
    from sqlalchemy.exc import IntegrityError
    from databoard.db_tools import add_event
    try:
        add_event(problem_name, event_name, event_title, is_public, force)
    except IntegrityError:
        logger.info(
            'Attempting to delete event, use "force=True" '
            'if you know what you are doing')


@task
def make_event_admin(c, event, user):
    from databoard.db_tools import make_event_admin

    make_event_admin(event_name=event, admin_name=user)


@task
def train_test(c, event, team=None, submission=None, state=None, force=False,
               is_save_y_pred=False, is_parallelize=True):
    from databoard.db_tools import (train_test_submissions,
                                    get_submissions,
                                    get_submissions_of_state)
    from databoard.config import sandbox_d_name

    if state is not None:
        submissions = get_submissions_of_state(state)
    else:
        submissions = get_submissions(
            event_name=event, team_name=team, submission_name=submission)
    submissions = [
        sub
        for sub in submissions
        if sub.name != sandbox_d_name
    ]
    train_test_submissions(
        submissions, force_retrain_test=force, is_parallelize=is_parallelize)
    compute_contributivity(c, event, is_save_y_pred=is_save_y_pred)


@task
def score_submission(c, event, team, submission, is_save_y_pred=False):
    from databoard.db_tools import score_submission, get_submissions

    submissions = get_submissions(
        event_name=event, team_name=team, submission_name=submission)
    score_submission(submissions[0])
    compute_contributivity(c, event, is_save_y_pred=is_save_y_pred)


@task
def set_n_submissions(c, event=None):
    from databoard.db_tools import set_n_submissions
    set_n_submissions(event)


@task
def compute_contributivity(c, event, is_save_y_pred=False):
    from databoard.db_tools import compute_contributivity
    from databoard.db_tools import compute_historical_contributivity
    from databoard.db_tools import set_n_submissions
    compute_contributivity(event, is_save_y_pred=is_save_y_pred)
    compute_historical_contributivity(event)
    set_n_submissions(event)


@task
def delete_submission(c, event, team, submission):
    from databoard.db_tools import delete_submission
    delete_submission(event_name=event, team_name=team,
                      submission_name=submission)


@task
def create_user(c, name, password, lastname, firstname, email,
                access_level='user', hidden_notes=''):
    from databoard.db_tools import create_user

    user = create_user(
        name, password, lastname, firstname,
        email, access_level, hidden_notes)

    return user


@task
def generate_single_password(c):
    from databoard.utils import generate_single_password
    print(generate_single_password())


@task
def generate_passwords(c, filename, password_filename):
    from databoard.utils import generate_passwords
    print(generate_passwords(filename, password_filename))


@task
def add_users_from_file(c, filename, password_filename):
    """Add users.

    Users whould be the same in the same order in the two files.
    """
    import pandas as pd
    from termcolor import colored
    from databoard.model import NameClashError
    from databoard.utils import remove_non_ascii
    from databoard.db_tools import create_user

    users_to_add = pd.read_csv(filename)
    passwords = pd.read_csv(password_filename)
    users_to_add['password'] = passwords['password']
    ids = []
    for _, user in users_to_add.iterrows():
        print(user)
        try:
            acces_level = user.access_level
        except AttributeError:
            acces_level = 'user'
        try:
            entry = create_user(
                remove_non_ascii(user.name),
                user.password,
                user.lastname,
                user.firstname,
                user.email,
                acces_level,
                user.hidden_notes
            )

            ids.append(entry.id)
        except NameClashError:
            print(colored(
                'user {}:{} already in database'.format(user.name, user.email),
                'red'))

    for id_ in ids:
        print(id_)


@task
def send_password_mail(c, user_name, password):
    """Update <user_name>'s password to <password> and mail it to him/her.

    Parameters
    ----------
    user_name : user name
    password : new password
    """
    from databoard.db_tools import send_password_mail
    send_password_mail(user_name, password)


@task
def send_password_mails(c, password_filename):
    """Update <name>'s password to <password>, read from <password_f_name>.

    Can be generated by `generate_passwords <generate_passwords>`.
    Parameters
    ----------
    password_f_name : a csv file with columns `name` and `password`
    """
    from databoard.db_tools import send_password_mails
    send_password_mails(password_filename)


@task
def sign_up_event_users_from_file(c, filename, event):
    import pandas as pd
    from termcolor import colored
    from databoard.model import DuplicateSubmissionError
    from databoard.utils import remove_non_ascii

    users_to_sign_up = pd.read_csv(filename)
    for _, user in users_to_sign_up.iterrows():
        username = remove_non_ascii(user.name)
        print('signing up {} to {}'.format(username, event))
        try:
            sign_up_team(c, event, username)
        except DuplicateSubmissionError:
            print(colored(
                'user {}:{} already signed up'.format(username, user.email),
                'red'))


@task
def update_leaderboards(c, event=None):
    from databoard.model import Event
    from databoard.db_tools import update_leaderboards
    if event is None:
        events = Event.query.all()
        for event in events:
            update_leaderboards(event.name)
    else:
        update_leaderboards(event)


@task
def update_user_leaderboards(c, event, user):
    from databoard.db_tools import update_user_leaderboards
    update_user_leaderboards(event, user)


@task
def update_all_user_leaderboards(c, event=None):
    from databoard.model import Event
    from databoard.db_tools import update_all_user_leaderboards
    if event is None:
        events = Event.query.all()
        for event in events:
            update_all_user_leaderboards(event.name)
    else:
        update_all_user_leaderboards(event)


@task
def backend_train_test_loop(c, event=None, timeout=30,
                            is_compute_contributivity=True,
                            is_parallelize=True):
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

    backend_train_test_loop(
        event, timeout, is_compute_contributivity, is_parallelize)


@task
def set_state(c, event, team, submission, state):
    "Set submission state"
    from databoard.db_tools import set_state
    set_state(event, team, submission, state)


@task
def exclude_from_ensemble(c, event, team, submission):
    from databoard.db_tools import exclude_from_ensemble
    exclude_from_ensemble(event, team, submission)
