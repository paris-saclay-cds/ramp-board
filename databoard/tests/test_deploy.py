from __future__ import print_function, absolute_import

import os

from rampdb.model import NameClashError

from databoard import ramp_data_path, ramp_kits_path
import databoard.db_tools as db_tools
from databoard.deploy import deploy


def test_deploy():
    deploy()


def test_add_users():
    db_tools.create_user(
        name='test_user', password='test',
        lastname='Test', firstname='User',
        email='test.user@gmail.com', access_level='user')
    db_tools.create_user(
        name='test_iris_admin', password='test',
        lastname='Admin', firstname='Iris',
        email='iris.admin@gmail.com', access_level='user')
    try:
        db_tools.create_user(
            name='test_user', password='test', lastname='Test',
            firstname='User', email='test.user2@gmail.com')
    except NameClashError as e:
        assert e.value == 'username is already in use'
    try:
        db_tools.create_user(
            name='test_user', password='test', lastname='Test',
            firstname='User', email='test.user@gmail.com')
    except NameClashError as e:
        assert e.value ==\
            'username is already in use and email is already in use'


def test_setup_workflows():
    db_tools.setup_workflows()


def _add_problem_and_event(problem_name, test_user_name):
    problem_kits_path = os.path.join(ramp_kits_path, problem_name)
    problem_data_path = os.path.join(ramp_data_path, problem_name)
    os.system('git clone https://github.com/ramp-data/{}.git {}'.format(
        problem_name, problem_data_path))
    os.system('git clone https://github.com/ramp-kits/{}.git {}'.format(
        problem_name, problem_kits_path))
    os.chdir(problem_data_path)
    print('Preparing {} data...'.format(problem_name))
    os.system('python prepare_data.py')
    os.chdir(problem_kits_path)
    os.system('jupyter nbconvert --to html {}_starting_kit.ipynb'.format(
        problem_name))

    db_tools.add_problem(
        problem_name, force=True)
    event_name = '{}_test'.format(problem_name)
    event_title = 'test event'
    db_tools.add_event(
        problem_name, event_name, event_title, is_public=True, force=True)
    db_tools.sign_up_team(event_name, test_user_name)
    db_tools.submit_starting_kit(event_name, test_user_name)
    submissions = db_tools.get_submissions(event_name, test_user_name)
    db_tools.train_test_submissions(
        submissions, force_retrain_test=True, is_parallelize=False)
    db_tools.compute_contributivity(event_name)
    db_tools.update_leaderboards(event_name)
    db_tools.update_user_leaderboards(event_name, test_user_name)
    db_tools.compute_contributivity(event_name)


def test_add_problem_and_event():
    _add_problem_and_event('iris', 'test_user')
    _add_problem_and_event('iris', 'test_user')
    _add_problem_and_event('boston_housing', 'test_user')


def test_make_event_admin():
    db_tools.make_event_admin('iris_test', 'test_iris_admin')


def test_add_keywords():
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


def test_leaderboard():
    print('***************** Leaderboard ****************')
    print(db_tools.get_leaderboards(
        'iris_test', user_name='test_user'))
    print(db_tools.get_leaderboards(
        'boston_housing_test', user_name='test_user'))
    print('***************** Private leaderboard ****************')
    print(db_tools.get_private_leaderboards('iris_test'))
    print(db_tools.get_private_leaderboards('boston_housing_test'))
    print('*********** Leaderboard of test_user ***********')
    print(db_tools.get_leaderboards(
        'iris_test', user_name='test_user'))
    print('*********** Private leaderboard of test_user ***********')
    print(db_tools.get_private_leaderboards(
        'iris_test', user_name='test_user'))
    print('*********** Failing leaderboard of test_user ***********')
    print(db_tools.get_failed_leaderboard('iris_test', user_name='test_user'))
    print('*********** New leaderboard of test_user ***********')
    print(db_tools.get_new_leaderboard('iris_test', user_name='test_user'))
