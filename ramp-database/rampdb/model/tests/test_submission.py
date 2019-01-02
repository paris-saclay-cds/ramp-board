import os
import re
import shutil

import numpy as np
import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampwf.prediction_types.base import BasePrediction

from rampdb.model import Event
from rampdb.model import EventScoreType
from rampdb.model import Model
from rampdb.model import SubmissionScore
from rampdb.model import Team

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.submission import get_submission_by_id


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture(scope='module')
def session_scope_module(config):
    try:
        create_toy_db(config)
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


def test_submission_model_property(session_scope_module):
    # check that the property of Submission
    submission = get_submission_by_id(session_scope_module, 5)
    assert (str(submission) ==
            'Submission(iris_test/test_user/starting_kit_test)')
    assert 'Submission(event_name=' in repr(submission)

    assert isinstance(submission.team, Team)
    assert isinstance(submission.event, Event)
    assert submission.official_score_name == 'acc'
    assert isinstance(submission.official_score, SubmissionScore)
    assert all([isinstance(score, EventScoreType)
                for score in submission.score_types])
    assert issubclass(submission.Predictions, BasePrediction)
    assert submission.is_not_sandbox is True
    assert submission.is_error is False
    assert submission.is_public_leaderboard is False
    assert submission.is_private_leaderboard is False
    assert (os.path.join('submissions', 'submission_000000005') in
            submission.path)
    assert submission.basename == 'submission_000000005'
    assert "submissions.submission_000000005" in submission.module
    assert len(submission.f_names) == 1
    assert submission.f_names[0] == 'classifier.py'
    assert submission.link == '/' + os.path.join(submission.hash_,
                                                 'classifier.py')
    assert re.match('<a href={}>{}/{}/{}</a>'
                    .format(submission.link, submission.event.name,
                            submission.team.name, submission.name),
                    submission.full_name_with_link)
    assert re.match('<a href={}>{}</a>'
                    .format(submission.link, submission.name),
                    submission.name_with_link)
    assert re.match('<a href=.*{}.*error.txt>{}</a>'
                    .format(submission.hash_, submission.state),
                    submission.state_with_link)

    for score in submission.ordered_scores(score_names=['acc', 'error']):
        assert isinstance(score, SubmissionScore)


def test_submission_model_set_state(session_scope_module):
    submission = get_submission_by_id(session_scope_module, 5)
    submission.set_state('scored')
    assert submission.state == 'scored'
    for cv_fold in submission.on_cv_folds:
        assert cv_fold.state == 'scored'


def test_submission_model_reset(session_scope_module):
    submission = get_submission_by_id(session_scope_module, 5)
    for score in submission.ordered_scores(score_names=['acc', 'error']):
        assert isinstance(score, SubmissionScore)
        # set the score to later test the reset function
        score.valid_score_cv_bag = 1.0
        score.test_score_cv_bag = 1.0
        score.valid_score_cv_bags = np.ones(2)
        score.test_score_cv_bags = np.ones(2)
    # set to non-default the variable that should change with reset
    submission.error_msg = 'simulate an error'
    submission.contributivity = 100.
    submission.reset()
    assert submission.contributivity == pytest.approx(0)
    assert submission.state == 'new'
    assert submission.error_msg == ''
    for score, worse_score in zip(submission.ordered_scores(['acc', 'error']),
                                  [0, 1]):
        assert score.valid_score_cv_bag == pytest.approx(worse_score)
        assert score.test_score_cv_bag == pytest.approx(worse_score)
        assert score.valid_score_cv_bags is None
        assert score.test_score_cv_bags is None


def test_submission_model_set_error(session_scope_module):
    submission = get_submission_by_id(session_scope_module, 5)
    error = 'training_error'
    error_msg = 'simulate an error'
    submission.set_error(error, error_msg)
    assert submission.state == error
    assert submission.error_msg == error_msg
    for cv_fold in submission.on_cv_folds:
        assert cv_fold.state == error
        assert cv_fold.error_msg == error_msg


@pytest.mark.parametrize(
    "state, expected_contributivity",
    [('scored', 0.3), ('training_error', 0.0)]
)
def test_submission_model_set_contributivity(session_scope_module, state,
                                             expected_contributivity):
    submission = get_submission_by_id(session_scope_module, 5)
    # set the state of the submission such that the contributivity
    submission.set_state(state)
    # set the fold contributivity to non-default
    for cv_fold in submission.on_cv_folds:
        cv_fold.contributivity = 0.3
    submission.set_contributivity()
    assert submission.contributivity == pytest.approx(expected_contributivity)


@pytest.mark.parametrize(
    "state_cv_folds, expected_state",
    [(['tested', 'tested'], 'tested'),
     (['tested', 'validated'], 'validated'),
     (['validated', 'validated'], 'validated'),
     (['trained', 'validated'], 'trained'),
     (['trained', 'tested'], 'trained'),
     (['trained', 'trained'], 'trained'),
     (['training_error', 'tested'], 'training_error'),
     (['validating_error', 'tested'], 'validating_error'),
     (['testing_error', 'tested'], 'testing_error')]
)
def test_submission_model_set_state_after_training(session_scope_module,
                                                   state_cv_folds,
                                                   expected_state):
    submission = get_submission_by_id(session_scope_module, 5)
    # set the state of the each fold
    for cv_fold, fold_state in zip(submission.on_cv_folds, state_cv_folds):
        cv_fold.state = fold_state
    submission.set_state_after_training()
    assert submission.state == expected_state


def test_submission_score_model_property(session_scope_module):
    # get the submission associated with the 5th submission (iris)
    # we get only the information linked to the accuracy score which the first
    # score
    submission_score = \
        (session_scope_module.query(SubmissionScore)
                             .filter(SubmissionScore.submission_id == 5)
                             .first())
    assert submission_score.score_name == 'acc'
    assert callable(submission_score.score_function)
    assert submission_score.precision == 2


@pytest.mark.parametrize(
    "step_score", ['train_score', 'valid_score', 'test_score']
)
def test_submission_score_model_scoring(session_scope_module, step_score):
    # get the submission associated with the 5th submission (iris)
    # we get only the information linked to the accuracy score which the first
    # score
    submission_score = \
        (session_scope_module.query(SubmissionScore)
                             .filter(SubmissionScore.submission_id == 5)
                             .first())
    # we set the score on the different fold to check the mean and std
    # computation on those folds.
    for cv_fold, fold_score in zip(submission_score.on_cv_folds,
                                   [0.2, 0.8]):
        setattr(cv_fold, step_score, fold_score)

    assert (getattr(submission_score, '{}_cv_mean'.format(step_score)) ==
            pytest.approx(0.5))
    assert (getattr(submission_score, '{}_cv_std'.format(step_score)) ==
            pytest.approx(0.3))
