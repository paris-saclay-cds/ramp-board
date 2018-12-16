import os
import subprocess
import shutil

from git import Repo

from . import db
from . import deployment_path
from . import ramp_config

from .db_tools import add_event
from .db_tools import add_problem
from .db_tools import approve_user
from .db_tools import create_user
from .db_tools import setup_files_extension_type
from .db_tools import sign_up_team


def create_test_db():
    """Create an empty test db and setup the files.

    The different settings should be set as environment variables. The
    environment variables to set are:

    * DATABOARD_STAGE: stage of the database. Need to be `'TESTING'` to use
      this function;
    * DATABOARD_DB_URL_TEST: database URL.
    """
    if os.getenv('DATABOARD_STAGE') in ['TEST', 'TESTING']:
        shutil.rmtree(deployment_path, ignore_errors=True)
        os.makedirs(deployment_path)
        os.makedirs(ramp_config['ramp_kits_path'])
        os.makedirs(ramp_config['ramp_data_path'])
        os.makedirs(ramp_config['ramp_submissions_path'])
        # create the empty database
        db.session.close()
        db.drop_all()
        db.create_all()
        setup_files_extension_type()
    else:
        raise AttributeError('DATABOARD_STAGE should be set to TESTING for '
                             '`deploy` to work')


# def create_toy_dataset():
#     """Create a toy dataset with couple of users, problems, events."""
#     create_test_db()
#     add_users()
#     add_problems()
#     add_events()
#     sign_up_team_to_events()


def add_users():
    """Add dummy users in the database."""
    create_user(
        name='test_user', password='test',
        lastname='Test', firstname='User',
        email='test.user@gmail.com', access_level='asked')
    approve_user('test_user')
    create_user(
        name='test_iris_admin', password='test',
        lastname='Admin', firstname='Iris',
        email='iris.admin@gmail.com', access_level='user')


def _setup_ramp_kits_ramp_data(problem_name):
    """Clone ramp-kits and ramp-data repository and setup it up."""
    # TODO: This function does not have a unit test but only used in
    # integration testing.
    problem_kits_path = os.path.join(ramp_config['ramp_kits_path'],
                                     problem_name)
    ramp_kits_url = 'https://github.com/ramp-kits/{}.git'.format(problem_name)
    ramp_kits_repo = Repo.clone_from(ramp_kits_url, problem_kits_path)

    problem_data_path = os.path.join(ramp_config['ramp_data_path'],
                                     problem_name)
    ramp_data_url = 'https://github.com/ramp-data/{}.git'.format(problem_name)
    ramp_data_repo = Repo.clone_from(ramp_data_url, problem_data_path)

    current_directory = os.getcwd()
    os.chdir(problem_data_path)
    subprocess.check_output(["python", "prepare_data.py"])
    os.chdir(problem_kits_path)
    # TODO: I don't think that we need to convert the notebook here. It should
    # not be used the related tests.
    subprocess.check_output(["jupyter", "nbconvert", "--to", "html",
                             "{}_starting_kit.ipynb".format(problem_name)])
    os.chdir(current_directory)


def add_problems():
    """Add dummy problems into the database."""
    problems = ['iris', 'boston_housing']
    for problem_name in problems:
        _setup_ramp_kits_ramp_data(problem_name)
        add_problem(problem_name)


def add_events():
    """Add events in the database.

    Notes
    -----
    Be aware that :func:`add_problems` needs to be called before.
    """
    problems = ['iris', 'boston_housing']
    for problem_name in problems:
        event_name = '{}_test'.format(problem_name)
        event_title = 'test event'
        add_event(problem_name=problem_name, event_name=event_name,
                  event_title=event_title, is_public=True, force=False)


def sign_up_team_to_events():
    """Sign up user to the events in the database.

    Notes
    -----
    Be aware that :func:`add_users()`, :func:`add_problems`,
    and :func:`add_events` need to be called before.
    """
    for event_name in ['iris_test', 'boston_housing_test']:
        sign_up_team(event_name, 'test_user')


# def _add_problem_and_event(problem_name, test_user_name):
#     db_tools.sign_up_team(event_name, test_user_name)
#     db_tools.submit_starting_kit(event_name, test_user_name)
#     db_tools.submit_starting_kit(event_name, test_user_name)
#     submissions = db_tools.get_submissions(event_name, test_user_name)
#     db_tools.train_test_submissions(
#         submissions, force_retrain_test=True, is_parallelize=False)
#     try:
#         db_tools.submit_starting_kit(event_name, test_user_name)
#     except DuplicateSubmissionError as e:
#         assert e.value == 'of team "{}" at event "{}" exists already'.format(
#             test_user_name, '{}_test'.format(problem_name))
#     db_tools.set_state(event_name, test_user_name, 'starting_kit_test', 'new')
#     db_tools.train_test_submissions(
#         submissions, force_retrain_test=True, is_parallelize=False)

#     db_tools.compute_contributivity(event_name)
#     db_tools.update_leaderboards(event_name)
#     db_tools.update_user_leaderboards(event_name, test_user_name)
#     db_tools.compute_contributivity(event_name)