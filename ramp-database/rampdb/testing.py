"""
The :mod:`rampdb.testing` module create facility functions to test the
tools and model of ``ramp-database``.
"""

import logging
import os
import shutil
import subprocess

from git import Repo

from ramputils import generate_ramp_config

from .utils import setup_db
from .utils import session_scope

from .model import Model

from .tools.database import add_extension
from .tools.database import add_submission_file_type
from .tools.database import add_submission_file_type_extension
from .tools.event import add_event
from .tools.event import add_keyword
from .tools.event import add_problem
from .tools.event import add_problem_keyword
from .tools.user import approve_user
from .tools.user import add_user
from .tools.team import sign_up_team
from .tools.submission import submit_starting_kits

logger = logging.getLogger('RAMP-DATABASE')


def create_test_db(database_config, ramp_config):
    """Create an empty test database and the setup the files for RAMP.

    Note: this will forcedly remove any existing content in the deployment
    directory.

    Parameters
    ----------
    database_config : dict
        The configuration file containing the database information.
    ramp_config : dict
        The configuration file containing the information about a RAMP event.
    """
    database_config = database_config['sqlalchemy']
    # we can automatically setup the database from the config file used for the
    # tests.
    ramp_config = generate_ramp_config(ramp_config)
    shutil.rmtree(ramp_config['deployment_dir'], ignore_errors=True)
    os.makedirs(ramp_config['ramp_kits_dir'])
    os.makedirs(ramp_config['ramp_data_dir'])
    os.makedirs(ramp_config['ramp_submissions_dir'])
    db, _ = setup_db(database_config)
    Model.metadata.drop_all(db)
    Model.metadata.create_all(db)
    with session_scope(database_config) as session:
        setup_files_extension_type(session)


def create_toy_db(database_config, ramp_config):
    """Create a toy dataset with couple of users, problems, events.

    Parameters
    ----------
    database_config : dict
        The configuration file containing the database information.
    ramp_config : dict
        The configuration file containing the information about a RAMP event.
    """
    create_test_db(database_config, ramp_config)
    with session_scope(database_config['sqlalchemy']) as session:
        setup_toy_db(session, ramp_config)


# Setup functions: functions used to setup the database initially
def setup_toy_db(session, ramp_config):
    """Only setup the database by adding some data.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    ramp_config : dict
        The configuration file containing the information about a RAMP event.
    """
    add_users(session)
    add_problems(session, ramp_config)
    add_events(session, ramp_config)
    sign_up_teams_to_events(session)
    submit_all_starting_kits(session, ramp_config)


def _delete_line_from_file(f_name, line_to_delete):
    with open(f_name, "r+") as f:
        lines = f.readlines()
        f.seek(0)
        for line in lines:
            if line != line_to_delete:
                f.write(line)
        f.truncate()


def setup_ramp_kits_ramp_data(ramp_config, problem_name, force=False):
    """Clone ramp-kits and ramp-data repository and setup it up.

    Parameters
    ----------
    ramp_config : dict
        The configuration file containing the information about a RAMP event.
    problem_name : str
        The name of the problem.
    force : bool, default is False
        Whether or not to overwrite the RAMP kit and data repositories if they
        already exists.
    """
    ramp_config = generate_ramp_config(ramp_config)
    problem_kits_path = os.path.join(ramp_config['ramp_kits_dir'],
                                     problem_name)
    if not os.path.exists(ramp_config['ramp_kits_dir']):
        os.makedirs(ramp_config['ramp_kits_dir'])
    if os.path.exists(problem_kits_path):
        if not force:
            raise ValueError(
                'The RAMP kit repository was previously cloned. To replace '
                'it, you need to set "force=True".'
            )
        shutil.rmtree(problem_kits_path, ignore_errors=True)
    ramp_kits_url = 'https://github.com/ramp-kits/{}.git'.format(problem_name)
    Repo.clone_from(ramp_kits_url, problem_kits_path)

    problem_data_path = os.path.join(ramp_config['ramp_data_dir'],
                                     problem_name)
    if not os.path.exists(ramp_config['ramp_data_dir']):
        os.makedirs(ramp_config['ramp_data_dir'])
    if os.path.exists(problem_data_path):
        if not force:
            raise ValueError(
                'The RAMP data repository was previously cloned. To replace '
                'it, you need to set "force=True".'
            )
        shutil.rmtree(problem_data_path, ignore_errors=True)
    ramp_data_url = 'https://github.com/ramp-data/{}.git'.format(problem_name)
    Repo.clone_from(ramp_data_url, problem_data_path)

    current_directory = os.getcwd()
    os.chdir(problem_data_path)
    subprocess.check_output(["python", "prepare_data.py"])
    os.chdir(problem_kits_path)
    subprocess.check_output(["jupyter", "nbconvert", "--to", "html",
                             "{}_starting_kit.ipynb".format(problem_name)])
    # delete this line since it trigger in the front-end
    # (try to open execute "custom.css".)
    _delete_line_from_file("{}_starting_kit.html".format(problem_name),
                           '<link rel="stylesheet" href="custom.css">\n')
    os.chdir(current_directory)


def setup_files_extension_type(session):
    """Setup the files' extensions and types.

    This function registers the file extensions and types. This function
    should be called after creating the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    """
    extension_names = ['py', 'R', 'txt', 'csv']
    for name in extension_names:
        add_extension(session, name)

    submission_file_types = [
        ('code', True, 10 ** 5),
        ('text', True, 10 ** 5),
        ('data', False, 10 ** 8)
    ]
    for name, is_editable, max_size in submission_file_types:
        add_submission_file_type(session, name, is_editable, max_size)

    submission_file_type_extensions = [
        ('code', 'py'),
        ('code', 'R'),
        ('text', 'txt'),
        ('data', 'csv')
    ]
    for type_name, extension_name in submission_file_type_extensions:
        add_submission_file_type_extension(session, type_name, extension_name)


# Add functions: functions to populate the database to obtain a toy dataset
def add_users(session):
    """Add dummy users in the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    """
    add_user(
        session, name='test_user', password='test',
        lastname='Test', firstname='User',
        email='test.user@gmail.com', access_level='asked')
    approve_user(session, 'test_user')
    add_user(
        session, name='test_user_2', password='test',
        lastname='Test_2', firstname='User_2',
        email='test.user.2@gmail.com', access_level='user')
    approve_user(session, 'test_user_2')
    add_user(
        session, name='test_iris_admin', password='test',
        lastname='Admin', firstname='Iris',
        email='iris.admin@gmail.com', access_level='admin')


def add_problems(session, ramp_config):
    """Add dummy problems into the database. In addition, we add couple of
    keyword.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    ramp_config : dict
        The configuration file containing the information about a RAMP event.
    """
    problems = ['iris', 'boston_housing']
    for problem_name in problems:
        setup_ramp_kits_ramp_data(ramp_config, problem_name)
        internal_ramp_config = generate_ramp_config(ramp_config)
        add_problem(session, problem_name,
                    internal_ramp_config['ramp_kits_dir'],
                    internal_ramp_config['ramp_data_dir'])
        add_keyword(session, problem_name, 'data_domain',
                    category='scientific data')
        add_problem_keyword(session, problem_name=problem_name,
                            keyword_name=problem_name)
        add_keyword(session, problem_name + '_theme', 'data_science_theme',
                    category='classification')
        add_problem_keyword(session, problem_name=problem_name,
                            keyword_name=problem_name + '_theme')


def add_events(session, ramp_config):
    """Add events in the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    ramp_config : dict
        The configuration file containing the information about a RAMP event.

    Notes
    -----
    Be aware that :func:`add_problems` needs to be called before.
    """
    ramp_config = generate_ramp_config(ramp_config)
    problems = ['iris', 'boston_housing']
    for problem_name in problems:
        event_name = '{}_test'.format(problem_name)
        event_title = 'test event'
        add_event(session, problem_name=problem_name, event_name=event_name,
                  event_title=event_title,
                  ramp_sandbox_name=ramp_config['sandbox_name'],
                  ramp_submissions_path=ramp_config['ramp_submissions_dir'],
                  is_public=True, force=False)


def sign_up_teams_to_events(session):
    """Sign up user to the events in the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.

    Notes
    -----
    Be aware that :func:`add_users`, :func:`add_problems`,
    and :func:`add_events` need to be called before.
    """
    for event_name in ['iris_test', 'boston_housing_test']:
        sign_up_team(session, event_name, 'test_user')
        sign_up_team(session, event_name, 'test_user_2')


def submit_all_starting_kits(session, ramp_config):
    """Submit all starting kits.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    ramp_config : dict
        The configuration file containing the information about a RAMP event.
    """
    ramp_config = generate_ramp_config(ramp_config)
    for event, event_name in zip(['iris', 'boston_housing'],
                                 ['iris_test', 'boston_housing_test']):
        path_submissions = os.path.join(
            ramp_config['ramp_kits_dir'], event, 'submissions'
        )
        submit_starting_kits(session, event_name, 'test_user',
                             path_submissions)
        submit_starting_kits(session, event_name, 'test_user_2',
                             path_submissions)
