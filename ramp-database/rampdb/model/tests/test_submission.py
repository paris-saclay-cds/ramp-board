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
