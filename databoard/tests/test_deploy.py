from __future__ import print_function, absolute_import

import os

import databoard.db_tools as db_tools
from databoard import ramp_data_path, ramp_kits_path
from databoard.deploy import deploy
from databoard.model import NameClashError, User, Event, Submission
from databoard.config import sandbox_d_name


def test_deploy():
    deploy()


def test_add_users():
    db_tools.create_user(
        name='test_user', password='test',
        lastname='Test', firstname='User',
        email='test.user@gmail.com', access_level='asked')
    db_tools.approve_user('test_user')
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
    db_tools.add_problem(
        problem_name)
    db_tools.add_problem(
        problem_name, force=True)
    event_name = '{}_test'.format(problem_name)
    event_title = 'test event'
    db_tools.add_event(
        problem_name, event_name, event_title, is_public=True, force=True)
    db_tools.add_event(
        problem_name, event_name, event_title, is_public=True)
    db_tools.add_event(
        problem_name, event_name, event_title, is_public=True, force=True)
    db_tools.sign_up_team(event_name, test_user_name)
    db_tools.submit_starting_kit(event_name, test_user_name)
    submissions = db_tools.get_submissions(event_name, test_user_name)
    db_tools.train_test_submissions(
        submissions, force_retrain_test=True, is_parallelize=False)
    db_tools.set_state(event_name, test_user_name, 'starting_kit_test', 'new')
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
    db_tools.exclude_from_ensemble(
        'boston_housing_test', 'test_user', 'random_forest_100')
    db_tools.compute_contributivity('boston_housing_test')
    db_tools.update_leaderboards('boston_housing_test')
    db_tools.update_user_leaderboards('boston_housing_test', 'test_user')
    db_tools.update_all_user_leaderboards('boston_housing_test')
    db_tools.update_all_user_leaderboards('iris_test')
    db_tools.delete_submission('iris_test', 'test_user', 'starting_kit_test')
    db_tools.compute_contributivity('iris_test')
    db_tools.update_leaderboards('iris_test')
    db_tools.update_user_leaderboards('iris_test', 'test_user')
    db_tools.set_n_submissions()
    db_tools.set_n_submissions('boston_housing_test')


def test_is_dot_dot_dot():
    event = Event.query.filter_by(name='boston_housing_test').one()
    user = User.query.filter_by(name='test_user').one()
    submission = Submission.query.filter_by(name='starting_kit_test').one()
    assert db_tools.is_user_signed_up('boston_housing_test', 'test_user')
    assert not db_tools.is_user_asked_sign_up(
        'boston_housing_test', 'test_user')
    assert not db_tools.is_admin(event, user)
    assert db_tools.is_public_event(event, user)
    assert db_tools.is_open_leaderboard(event, user)
    assert db_tools.is_open_code(event, user, submission)


def test_make_event_admin():
    db_tools.make_event_admin('iris_test', 'test_iris_admin')


def test_add_keywords():
    db_tools.add_keyword('botany', 'data_domain', 'scientific data', 'Botany')
    db_tools.add_keyword('botany', 'data_domain', 'scientific data', 'Botany.')
    db_tools.add_keyword(
        'botany', 'data_domain', 'scientific data', 'Botany.', force=True)
    db_tools.add_keyword(
        'real estate', 'data_domain', 'industrial data', 'Real estate.')
    db_tools.add_keyword(
        'regression', 'data_science_theme', None, 'Regression.')
    db_tools.add_keyword(
        'classification', 'data_science_theme', None, 'Classification.')
    db_tools.add_problem_keyword('iris', 'classification')
    db_tools.add_problem_keyword('iris', 'classification')
    db_tools.add_problem_keyword('iris', 'classification', force=True)
    db_tools.add_problem_keyword('iris', 'botany')
    db_tools.add_problem_keyword('boston_housing', 'regression')
    db_tools.add_problem_keyword('boston_housing', 'real estate')


def test_update_profile():
    class Field():
        def __init__(self, data):
            self.data = data

    class UserUpdateProfileForm():
        def __init__(self):
            self.user_name = Field('test_user')
            self.lastname = Field('test')
            self.firstname = Field('test')
            self.email = Field('test.user@gmail.com')
            self.linkedin_url = Field('test')
            self.twitter_url = Field('test')
            self.facebook_url = Field('test')
            self.google_url = Field('test')
            self.github_url = Field('test')
            self.website_url = Field('test')
            self.bio = Field('test')
            self.is_want_news = Field(False)

    user = User.query.filter_by(name='test_user').one()
    form = UserUpdateProfileForm()
    db_tools.update_user(user, form)
    form.email = Field('iris.admin@gmail.com')
    try:
        db_tools.update_user(user, form)
    except NameClashError as e:
        assert e.value == 'email is already in use'


def test_get_sandbox():
    event = Event.query.filter_by(name='boston_housing_test').one()
    user = User.query.filter_by(email='test.user@gmail.com').one()
    sandbox = db_tools.get_sandbox(event, user)
    assert sandbox.name == sandbox_d_name


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
