from __future__ import absolute_import

import os
import shutil

import datetime
import pytest
from git.exc import GitCommandError

from rampdb.model import Problem
from rampdb.model import User
from rampdb.model import NameClashError

from databoard import db
from databoard import deployment_path
from databoard import ramp_config

import databoard.db_tools as db_tools

from databoard.testing import add_events
from databoard.testing import add_problems
from databoard.testing import add_users
from databoard.testing import create_test_db
from databoard.testing import sign_up_teams_to_events
from databoard.testing import submit_all_starting_kits


@pytest.fixture
def setup_db():
    try:
        create_test_db()
        yield
    finally:
        shutil.rmtree(deployment_path, ignore_errors=True)
        db.session.close()
        db.session.remove()
        db.drop_all()


def test_add_users(setup_db):
    add_users()
    users = db.session.query(User).all()
    for user in users:
        assert user.name in ('test_user', 'test_user_2', 'test_iris_admin')
    err_msg = 'username is already in use'
    with pytest.raises(NameClashError, match=err_msg):
        add_users()


def test_add_problems(setup_db):
    add_problems()
    problems = db.session.query(Problem).all()
    for problem in problems:
        assert problem.name in ('iris', 'boston_housing')
    # trying to add twice the same problem will raise a git error since the
    #  repositories already exist.
    with pytest.raises(GitCommandError):
        add_problems()


def test_add_events(setup_db):
    add_problems()
    add_events()
    with pytest.raises(ValueError):
        add_events()


def test_sign_up_team_to_events(setup_db):
    add_users()
    add_problems()
    add_events()
    sign_up_teams_to_events()


def test_submit_all_starting_kits(setup_db):
    add_users()
    add_problems()
    add_events()
    sign_up_teams_to_events()
    submit_all_starting_kits()


# def _add_problem_and_event(problem_name, test_user_name):
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


# def test_add_problem_and_event():
#     _add_problem_and_event('iris', 'test_user')
#     _add_problem_and_event('iris', 'test_user')
#     _add_problem_and_event('boston_housing', 'test_user')


# def test_add_submission_similarity():
#     u = User.query.filter_by(name='test_user').one()
#     t = Team.query.filter_by(name='test_user').one()
#     e = Event.query.filter_by(name='boston_housing_test').one()
#     et = EventTeam.query.filter_by(event=e, team=t).one()
#     submissions = Submission.query.filter_by(event_team=et).all()
#     source_submission = submissions[0]
#     target_submission = submissions[1]
#     db_tools.add_submission_similarity(
#         type='target_credit', user=u, source_submission=source_submission,
#         target_submission=target_submission, similarity=0.8,
#         timestamp=datetime.datetime.utcnow())
#     db_tools.add_user_interaction(
#         interaction='giving credit', user=u, event=e, ip='0.0.0.0',
#         submission=target_submission)


# def test_is_dot_dot_dot():
#     event = Event.query.filter_by(name='boston_housing_test').one()
#     user = User.query.filter_by(name='test_user').one()
#     submission = Submission.query.filter_by(name='starting_kit_test').all()[0]
#     assert db_tools.is_user_signed_up('boston_housing_test', 'test_user')
#     assert not db_tools.is_user_asked_sign_up(
#         'boston_housing_test', 'test_user')
#     assert not db_tools.is_admin(event, user)
#     assert db_tools.is_public_event(event, user)
#     assert db_tools.is_open_leaderboard(event, user)
#     assert db_tools.is_open_code(event, user, submission)


# def test_make_event_admin():
#     db_tools.make_event_admin('iris_test', 'test_iris_admin')


# def test_add_keywords():
#     db_tools.add_keyword('botany', 'data_domain', 'scientific data', 'Botany')
#     db_tools.add_keyword('botany', 'data_domain', 'scientific data', 'Botany.')
#     db_tools.add_keyword(
#         'botany', 'data_domain', 'scientific data', 'Botany.', force=True)
#     db_tools.add_keyword(
#         'real estate', 'data_domain', 'industrial data', 'Real estate.')
#     db_tools.add_keyword(
#         'regression', 'data_science_theme', None, 'Regression.')
#     db_tools.add_keyword(
#         'classification', 'data_science_theme', None, 'Classification.')
#     db_tools.add_problem_keyword('iris', 'classification')
#     db_tools.add_problem_keyword('iris', 'classification')
#     db_tools.add_problem_keyword('iris', 'classification', force=True)
#     db_tools.add_problem_keyword('iris', 'botany')
#     db_tools.add_problem_keyword('boston_housing', 'regression')
#     db_tools.add_problem_keyword('boston_housing', 'real estate')


# def test_update_profile():
#     class Field():
#         def __init__(self, data):
#             self.data = data

#     class UserUpdateProfileForm():
#         def __init__(self):
#             self.user_name = Field('test_user')
#             self.lastname = Field('test')
#             self.firstname = Field('test')
#             self.email = Field('test.user@gmail.com')
#             self.linkedin_url = Field('test')
#             self.twitter_url = Field('test')
#             self.facebook_url = Field('test')
#             self.google_url = Field('test')
#             self.github_url = Field('test')
#             self.website_url = Field('test')
#             self.bio = Field('test')
#             self.is_want_news = Field(False)

#     user = User.query.filter_by(name='test_user').one()
#     form = UserUpdateProfileForm()
#     db_tools.update_user(user, form)
#     form.email = Field('iris.admin@gmail.com')
#     try:
#         db_tools.update_user(user, form)
#     except NameClashError as e:
#         assert e.value == 'email is already in use'


# def test_get_sandbox():
#     event = Event.query.filter_by(name='boston_housing_test').one()
#     user = User.query.filter_by(email='test.user@gmail.com').one()
#     sandbox = db_tools.get_sandbox(event, user)
#     assert sandbox.name == ramp_config['sandbox_dir']


# def test_get_earliest_new_submission():
#     db_tools.get_earliest_new_submission()
#     db_tools.get_earliest_new_submission('iris_test')
#     db_tools.get_earliest_new_submission('not_iris_test')


# def test_leaderboard():
#     print('***************** Leaderboard ****************')
#     print(db_tools.get_leaderboards(
#         'iris_test', user_name='test_user'))
#     print(db_tools.get_leaderboards(
#         'boston_housing_test', user_name='test_user'))
#     print('***************** Private leaderboard ****************')
#     print(db_tools.get_private_leaderboards('iris_test'))
#     print(db_tools.get_private_leaderboards('boston_housing_test'))
#     print('*********** Leaderboard of test_user ***********')
#     print(db_tools.get_leaderboards(
#         'iris_test', user_name='test_user'))
#     print('*********** Private leaderboard of test_user ***********')
#     print(db_tools.get_private_leaderboards(
#         'iris_test', user_name='test_user'))
#     print('*********** Failing leaderboard of test_user ***********')
#     print(db_tools.get_failed_leaderboard('iris_test', user_name='test_user'))
#     print('*********** New leaderboard of test_user ***********')
#     print(db_tools.get_new_leaderboard('iris_test', user_name='test_user'))


# def test_model():
#     u = User.query.filter_by(name='test_user').one()
#     u.is_active
#     u.is_anonymous
#     u.get_id()
#     u.__repr__()

#     p = Problem.query.filter_by(name='iris').one()
#     p.__repr__()
#     p.title
#     p.module
#     p.Predictions
#     p.get_train_data()
#     p.get_test_data()
#     predictions = p.ground_truths_train()
#     p.ground_truths_test()
#     p.ground_truths_valid(range(len(predictions.y_pred)))
#     p.workflow_object

#     for e in p.events:
#         e.workflow
#         e.combined_combined_valid_score_str
#         e.combined_combined_test_score_str
#         e.combined_foldwise_valid_score_str
#         e.combined_foldwise_test_score_str
#         e.is_open
#         e.is_public_open
#         e.is_closed
#         e.n_jobs
#         e.n_participants
#         for st in e.score_types:
#             st.__repr__()
#             st.score_type_object
#             st.score_function
#             st.is_lower_the_better
#             st.minimum
#             st.maximum
#             st.worst
#         for cv in e.cv_folds:
#             e.__repr__()
#         for et in e.event_teams:
#             et.__repr__()

#     w = p.workflow
#     w.__repr__()
#     for we in w.elements:
#         we.__repr__()
#         we.type
#         we.file_type
#         we.is_editable
#         we.max_size
#         wet = we.workflow_element_type
#         wet.__repr__()
#         wet.file_type
#         wet.is_editable
#         wet.max_size
#         sft = wet.type
#         for e in sft.extensions:
#             e.file_type
#             e.extension_name
#         for sf in we.submission_files:
#             sf.is_editable
#             sf.extension
#             sf.type
#             sf.name
#             sf.f_name
#             sf.link
#             sf.path
#             sf.name_with_link
#             code = sf.get_code()
#             sf.set_code(code)
#             sf.__repr__()

#     ss = Submission.query.all()
#     ss = [s for s in ss if s.is_not_sandbox and not s.is_error]
#     for s in ss:
#         print(s.event, s.name)
#         for score in s.scores:
#             score.score_name
#             score.score_function
#             score.precision
#             score.train_score_cv_mean
#             score.valid_score_cv_mean
#             score.test_score_cv_mean
#             score.train_score_cv_std
#             score.valid_score_cv_std
#             score.test_score_cv_std
#             score.score_name
#             score.score_name
#             for cv_score in score.on_cv_folds:
#                 cv_score.name
#                 cv_score.event_score_type
#                 cv_score.score_function
#         for cv in s.on_cv_folds:
#             cv.__repr__()
#             cv.is_public_leaderboard
#             cv.is_trained
#             cv.is_validated
#             cv.is_tested
#             cv.is_error
#             cv.full_train_predictions
#             cv.train_predictions
#             cv.valid_predictions
#             cv.test_predictions
#             cv.official_score
#             cv.reset()
#             cv.compute_train_scores()
#             cv.compute_valid_scores()
#             cv.compute_test_scores()
#         s.__str__()
#         s.__repr__()
#         s.team
#         s.event
#         s.official_score_function
#         s.official_score_name
#         s.official_score
#         s.score_types
#         s.Predictions
#         s.is_not_sandbox
#         s.is_error
#         s.is_public_leaderboard
#         s.is_private_leaderboard
#         s.path
#         s.module
#         s.f_names
#         s.link
#         s.full_name_with_link
#         s.name_with_link
#         s.state_with_link
#         s.ordered_scores
#         s.set_state(s.state)
#         s.set_error('training_error', 'error message')
#         s.set_contributivity()
#         s.reset()

#     sss = SubmissionSimilarity.query.all()
#     for ss in sss:
#         ss.__repr__()

#     uis = UserInteraction.query.all()
#     for ui in uis:
#         ui.__repr__()
#         ui.submission_file_diff_link
#         ui.event
#         ui.team


# def test_delete_submissions():
#     db_tools.exclude_from_ensemble(
#         'boston_housing_test', 'test_user', 'random_forest_100')
#     db_tools.compute_contributivity('boston_housing_test')
#     db_tools.update_leaderboards('boston_housing_test')
#     db_tools.update_user_leaderboards('boston_housing_test', 'test_user')
#     db_tools.update_all_user_leaderboards('boston_housing_test')
#     db_tools.update_all_user_leaderboards('iris_test')
#     db_tools.delete_submission('iris_test', 'test_user', 'starting_kit_test')
#     db_tools.compute_contributivity('iris_test')
#     db_tools.update_leaderboards('iris_test')
#     db_tools.update_user_leaderboards('iris_test', 'test_user')
#     db_tools.set_n_submissions()
#     db_tools.set_n_submissions('boston_housing_test')
