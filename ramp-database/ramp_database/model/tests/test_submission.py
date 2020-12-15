import os
import re
import shutil
from datetime import datetime

import numpy as np
from numpy.testing import assert_allclose
import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from rampwf.prediction_types.base import BasePrediction

from ramp_database.model import DetachedSubmissionOnCVFold
from ramp_database.model import Event
from ramp_database.model import EventScoreType
from ramp_database.model import Extension
from ramp_database.model import HistoricalContributivity
from ramp_database.model import Model
from ramp_database.model import Submission
from ramp_database.model import SubmissionFile
from ramp_database.model import SubmissionFileType
from ramp_database.model import SubmissionFileTypeExtension
from ramp_database.model import SubmissionOnCVFold
from ramp_database.model import SubmissionScore
from ramp_database.model import SubmissionScoreOnCVFold
from ramp_database.model import Team
from ramp_database.model import WorkflowElementType

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.submission import get_submission_by_id

# id for the test submission with user=test_user and
# event=iris_test
ID_SUBMISSION = 7


@pytest.fixture(scope='module')
def session_scope_module(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_submission_model_property(session_scope_module):
    # check that the property of Submission
    submission = get_submission_by_id(session_scope_module, ID_SUBMISSION)
    assert re.match(r'Submission\(iris_test/test_user/.*\)',
                    str(submission))
    assert re.match(r'Submission\(event_name.*\)', repr(submission))

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
    submission_str = 'submission_00000000' + str(ID_SUBMISSION)
    assert (os.path.join('submissions', submission_str)
            in submission.path)
    assert submission.basename == submission_str
    assert "submissions." + submission_str in submission.module
    assert len(submission.f_names) == 1
    assert submission.f_names[0] == 'estimator.py'
    assert submission.link == '/' + os.path.join(submission.hash_,
                                                 'estimator.py')
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
    submission = get_submission_by_id(session_scope_module, ID_SUBMISSION)
    submission.set_state('sent_to_training')
    assert submission.state == 'sent_to_training'
    assert (
        submission.sent_to_training_timestamp - datetime.utcnow()
    ).total_seconds() < 10

    submission.set_state('training')
    assert submission.state == 'training'
    assert (
        submission.training_timestamp - datetime.utcnow()
    ).total_seconds() < 10

    submission.set_state('scored')
    assert submission.state == 'scored'
    for cv_fold in submission.on_cv_folds:
        assert cv_fold.state == 'scored'


def test_submission_model_reset(session_scope_module):
    submission = get_submission_by_id(session_scope_module, ID_SUBMISSION)
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
    submission = get_submission_by_id(session_scope_module, ID_SUBMISSION)
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
    submission = get_submission_by_id(session_scope_module, ID_SUBMISSION)
    # set the state of the submission such that the contributivity
    submission.set_state(state)
    # set the fold contributivity to non-default
    for cv_fold in submission.on_cv_folds:
        cv_fold.contributivity = 0.3
    submission.set_contributivity()
    assert submission.contributivity == pytest.approx(expected_contributivity)


@pytest.mark.parametrize(
    'backref, expected_type',
    [('historical_contributivitys', HistoricalContributivity),
     ('scores', SubmissionScore),
     ('files', SubmissionFile),
     ('on_cv_folds', SubmissionOnCVFold),
     ('sources', Submission),
     ('targets', Submission)]
)
def test_submission_model_backref(session_scope_module, backref,
                                  expected_type):
    submission = get_submission_by_id(session_scope_module, ID_SUBMISSION)
    backref_attr = getattr(submission, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)


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
    submission = get_submission_by_id(session_scope_module, ID_SUBMISSION)
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
                             .filter(
                                 SubmissionScore.submission_id == ID_SUBMISSION
                                 )
                             .first())
    assert submission_score.score_name == 'acc'
    assert callable(submission_score.score_function)
    assert submission_score.precision == 2


@pytest.mark.parametrize(
    'backref, expected_type',
    [('on_cv_folds', SubmissionScoreOnCVFold)]
)
def test_submission_score_model_backref(session_scope_module, backref,
                                        expected_type):
    submission_score = \
        (session_scope_module.query(SubmissionScore)
                             .filter(
                                 SubmissionScore.submission_id == ID_SUBMISSION
                                 )
                             .first())
    backref_attr = getattr(submission_score, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)


def test_submission_file_model_property(session_scope_module):
    # get the submission file of an iris submission with only a estimator file
    submission_file = \
        (session_scope_module.query(SubmissionFile)
                             .filter(
                                 SubmissionFile.submission_id == ID_SUBMISSION
                                 )
                             .first())
    assert re.match('SubmissionFile(.*)',
                    repr(submission_file))
    assert submission_file.is_editable is True
    assert submission_file.extension == 'py'
    assert submission_file.type == 'estimator'
    assert submission_file.name == 'estimator'
    assert submission_file.f_name == 'estimator.py'
    assert re.match('/.*estimator.py', submission_file.link)
    submission_name = 'submission_00000000' + str(ID_SUBMISSION)
    assert re.match(f'.*submissions.*{submission_name}.*estimator.py',
                    submission_file.path)
    assert re.match('<a href=".*estimator.py">.*estimator</a>',
                    submission_file.name_with_link)
    assert re.match('from sklearn.ensemble import RandomForestClassifier.*',
                    submission_file.get_code())
    submission_file.set_code(code='# overwriting a code file')
    assert submission_file.get_code() == '# overwriting a code file'


def test_submission_file_type_extension_model_property(session_scope_module):
    submission_file_type_extension = \
        (session_scope_module.query(SubmissionFileTypeExtension).first())
    assert submission_file_type_extension.file_type == 'code'
    assert submission_file_type_extension.extension_name == 'py'


@pytest.mark.parametrize(
    'backref, expected_type',
    [('submission_files', SubmissionFile)]
)
def test_submission_file_type_extension_model_backref(session_scope_module,
                                                      backref, expected_type):
    submission_file_type_extension = \
        (session_scope_module.query(SubmissionFileTypeExtension).first())
    backref_attr = getattr(submission_file_type_extension, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)


def test_submission_score_on_cv_fold_model_property(session_scope_module):
    cv_fold_score = (session_scope_module
        .query(SubmissionScoreOnCVFold)                         # noqa
        .filter(SubmissionScoreOnCVFold.submission_score_id ==  # noqa
                SubmissionScore.id)                             # noqa
        .filter(SubmissionScore.event_score_type_id ==          # noqa
                EventScoreType.id)                              # noqa
        .filter(SubmissionScore.submission_id == ID_SUBMISSION)  # noqa
        .filter(EventScoreType.name == 'acc')                   # noqa
        .first())                                               # noqa
    assert cv_fold_score.name == 'acc'
    assert isinstance(cv_fold_score.event_score_type, EventScoreType)
    assert callable(cv_fold_score.score_function)


def test_submission_on_cv_fold_model_property(session_scope_module):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(
                                 SubmissionOnCVFold.submission_id ==
                                 ID_SUBMISSION
                             ).first())
    cv_fold.state = 'scored'
    cv_fold.contributivity = 0.2
    assert repr(cv_fold) == 'state = scored, c = 0.2, best = False'
    assert isinstance(cv_fold.official_score, SubmissionScoreOnCVFold)
    assert cv_fold.official_score.name == 'acc'


@pytest.mark.parametrize(
    "state_set, expected_state",
    [('new', False),
     ('checked', False),
     ('checking_error', False),
     ('trained', False),
     ('training_error', False),
     ('validated', False),
     ('validating_error', False),
     ('tested', False),
     ('testing_error', False),
     ('training', False),
     ('sent_to_training', False),
     ('scored', True)]
)
def test_submission_on_cv_fold_model_is_public_leaderboard(
        session_scope_module, state_set, expected_state):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(
                                 SubmissionOnCVFold.submission_id ==
                                 ID_SUBMISSION
                             ).first())
    cv_fold.state = state_set
    assert cv_fold.is_public_leaderboard is expected_state


@pytest.mark.parametrize(
    "state_set, expected_state",
    [('new', False),
     ('checked', False),
     ('checking_error', False),
     ('trained', True),
     ('training_error', False),
     ('validated', True),
     ('validating_error', True),
     ('tested', True),
     ('testing_error', True),
     ('training', False),
     ('sent_to_training', False),
     ('scored', True)]
)
def test_submission_on_cv_fold_model_is_trained(session_scope_module,
                                                state_set, expected_state):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(
                                 SubmissionOnCVFold.submission_id ==
                                 ID_SUBMISSION
                             ).first())
    cv_fold.state = state_set
    assert cv_fold.is_trained is expected_state


@pytest.mark.parametrize(
    "state_set, expected_state",
    [('new', False),
     ('checked', False),
     ('checking_error', False),
     ('trained', False),
     ('training_error', False),
     ('validated', True),
     ('validating_error', False),
     ('tested', True),
     ('testing_error', True),
     ('training', False),
     ('sent_to_training', False),
     ('scored', True)]
)
def test_submission_on_cv_fold_model_is_validated(session_scope_module,
                                                  state_set, expected_state):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(
                                 SubmissionOnCVFold.submission_id ==
                                 ID_SUBMISSION
                             ).first())
    cv_fold.state = state_set
    assert cv_fold.is_validated is expected_state


@pytest.mark.parametrize(
    "state_set, expected_state",
    [('new', False),
     ('checked', False),
     ('checking_error', False),
     ('trained', False),
     ('training_error', False),
     ('validated', False),
     ('validating_error', False),
     ('tested', True),
     ('testing_error', False),
     ('training', False),
     ('sent_to_training', False),
     ('scored', True)]
)
def test_submission_on_cv_fold_model_is_tested(session_scope_module,
                                               state_set, expected_state):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(
                                 SubmissionOnCVFold.submission_id ==
                                 ID_SUBMISSION
                             ).first())
    cv_fold.state = state_set
    assert cv_fold.is_tested is expected_state


@pytest.mark.parametrize(
    "state_set, expected_state",
    [('new', False),
     ('checked', False),
     ('checking_error', True),
     ('trained', False),
     ('training_error', True),
     ('validated', False),
     ('validating_error', True),
     ('tested', False),
     ('testing_error', True),
     ('training', False),
     ('sent_to_training', False),
     ('scored', False)]
)
def test_submission_on_cv_fold_model_is_error(session_scope_module,
                                              state_set, expected_state):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(
                                 SubmissionOnCVFold.submission_id ==
                                 ID_SUBMISSION
                             ).first())
    cv_fold.state = state_set
    assert cv_fold.is_error is expected_state


def test_submission_on_cv_fold_model_predictions(session_scope_module):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(SubmissionOnCVFold.submission_id ==
                                     ID_SUBMISSION)
                             .first())
    # Set fake predictions to check the prediction properties
    cv_fold.full_train_y_pred = np.empty((120, 3))
    cv_fold.full_train_y_pred[:, 0] = 1
    cv_fold.full_train_y_pred[:, 1:] = 0
    cv_fold.test_y_pred = np.empty((30, 3))
    cv_fold.test_y_pred[:, 0] = 1
    cv_fold.test_y_pred[:, 1:] = 0
    assert isinstance(cv_fold.full_train_predictions, BasePrediction)
    assert isinstance(cv_fold.train_predictions, BasePrediction)
    assert isinstance(cv_fold.valid_predictions, BasePrediction)
    assert isinstance(cv_fold.test_predictions, BasePrediction)


def test_submission_on_cv_fold_model_reset(session_scope_module):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(SubmissionOnCVFold.submission_id ==
                                     ID_SUBMISSION)
                             .first())
    # set to non-default values]
    cv_fold.full_train_y_pred = np.empty((120, 3))
    cv_fold.full_train_y_pred[:, 0] = 1
    cv_fold.full_train_y_pred[:, 1:] = 0
    cv_fold.test_y_pred = np.ones(30)
    cv_fold.test_y_pred = np.empty((30, 3))
    cv_fold.test_y_pred[:, 0] = 1
    cv_fold.test_y_pred[:, 1:] = 0
    cv_fold.contributivity = 0.3
    cv_fold.best = True
    cv_fold.train_time = 1
    cv_fold.valid_time = 1
    cv_fold.test_time = 1
    cv_fold.state = 'scored'
    cv_fold.error_msg = 'simulate a message'
    for score in cv_fold.scores:
        if score.name == 'acc':
            score.train_score = 1.0
            score.valid_score = 1.0
            score.test_score = 1.0

    cv_fold.reset()
    assert cv_fold.contributivity == pytest.approx(0)
    assert cv_fold.best is False
    assert cv_fold.full_train_y_pred is None
    assert cv_fold.test_y_pred is None
    assert cv_fold.train_time == pytest.approx(0)
    assert cv_fold.valid_time == pytest.approx(0)
    assert cv_fold.test_time == pytest.approx(0)
    assert cv_fold.state == 'new'
    assert cv_fold.error_msg == ''
    for score in cv_fold.scores:
        if score.name == 'acc':
            assert score.train_score == pytest.approx(0)
            assert score.valid_score == pytest.approx(0)
            assert score.test_score == pytest.approx(0)


def test_submission_on_cv_fold_model_error(session_scope_module):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(SubmissionOnCVFold.submission_id ==
                                     ID_SUBMISSION)
                             .first())
    error = 'training_error'
    error_msg = 'simulate an error'
    cv_fold.set_error(error, error_msg)
    assert cv_fold.state == error
    assert cv_fold.error_msg == error_msg


@pytest.mark.filterwarnings('ignore:F-score is ill-defined and being set to')
def test_submission_on_cv_fold_model_train_scores(session_scope_module):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(SubmissionOnCVFold.submission_id ==
                                     ID_SUBMISSION)
                             .first())
    # Set fake predictions to compute the score
    cv_fold.state = 'trained'
    cv_fold.full_train_y_pred = np.empty((120, 3))
    cv_fold.full_train_y_pred[:, 0] = 1
    cv_fold.full_train_y_pred[:, 1:] = 0
    cv_fold.compute_train_scores()
    for score in cv_fold.scores:
        if score.name == 'acc':
            assert score.train_score == pytest.approx(0.3333333333333333)

    # simulate that the training did not complete
    cv_fold.state = 'training'
    cv_fold.compute_train_scores()
    for score in cv_fold.scores:
        if score.name == 'acc':
            assert score.train_score == pytest.approx(0)


@pytest.mark.filterwarnings('ignore:F-score is ill-defined and being set to')
def test_submission_on_cv_fold_model_valid_scores(session_scope_module):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(SubmissionOnCVFold.submission_id ==
                                     ID_SUBMISSION)
                             .first())
    # Set fake predictions to compute the score
    cv_fold.state = 'validated'
    cv_fold.full_train_y_pred = np.empty((120, 3))
    cv_fold.full_train_y_pred[:, 0] = 1
    cv_fold.full_train_y_pred[:, 1:] = 0
    cv_fold.compute_valid_scores()
    for score in cv_fold.scores:
        if score.name == 'acc':
            assert score.valid_score == pytest.approx(0.3333333333333333)

    # simulate that the training did not complete
    cv_fold.state = 'training'
    cv_fold.compute_valid_scores()
    for score in cv_fold.scores:
        if score.name == 'acc':
            assert score.valid_score == pytest.approx(0)


@pytest.mark.filterwarnings('ignore:F-score is ill-defined and being set to')
def test_submission_on_cv_fold_model_test_scores(session_scope_module):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(SubmissionOnCVFold.submission_id ==
                                     ID_SUBMISSION)
                             .first())
    # Set fake predictions to compute the score
    cv_fold.state = 'scored'
    cv_fold.test_y_pred = np.empty((30, 3))
    cv_fold.test_y_pred[:, 0] = 1
    cv_fold.test_y_pred[:, 1:] = 0
    cv_fold.compute_test_scores()
    for score in cv_fold.scores:
        if score.name == 'acc':
            assert score.test_score == pytest.approx(0.3333333333333333)

    # simulate that the training did not complete
    cv_fold.state = 'training'
    cv_fold.compute_test_scores()
    for score in cv_fold.scores:
        if score.name == 'acc':
            assert score.test_score == pytest.approx(0)


def test_submission_on_cv_fold_model_update(session_scope_module):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(SubmissionOnCVFold.submission_id ==
                                     ID_SUBMISSION)
                             .first())

    detached_cv_fold = DetachedSubmissionOnCVFold(cv_fold)
    detached_cv_fold.state = 'scored'
    detached_cv_fold.train_time = 1
    detached_cv_fold.valid_time = 2
    detached_cv_fold.full_train_y_pred = np.zeros((120, 3))
    detached_cv_fold.test_time = 3
    detached_cv_fold.test_y_pred = np.zeros((30, 3))

    cv_fold.update(detached_cv_fold)
    assert cv_fold.state == 'scored'
    assert cv_fold.train_time == 1
    assert cv_fold.valid_time == 2
    assert cv_fold.test_time == 3
    assert_allclose(cv_fold.full_train_y_pred, np.zeros((120, 3)))
    assert_allclose(cv_fold.test_y_pred, np.zeros((30, 3)))


@pytest.mark.parametrize(
    'backref, expected_type',
    [('scores', SubmissionScoreOnCVFold)]
)
def test_submission_on_cv_fold_model_backref(session_scope_module, backref,
                                             expected_type):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(SubmissionOnCVFold.submission_id ==
                                     ID_SUBMISSION)
                             .first())
    backref_attr = getattr(cv_fold, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)


def test_detached_submission_on_cv_fold_model(session_scope_module):
    cv_fold = \
        (session_scope_module.query(SubmissionOnCVFold)
                             .filter(SubmissionOnCVFold.submission_id ==
                                     ID_SUBMISSION)
                             .first())

    detached_cv_fold = DetachedSubmissionOnCVFold(cv_fold)
    assert re.match('Submission(.*).*', repr(detached_cv_fold))


@pytest.mark.parametrize(
    'backref, expected_type',
    [('submission_file_types', SubmissionFileTypeExtension)]
)
def test_extension_model_backref(session_scope_module, backref, expected_type):
    extension = session_scope_module.query(Extension).first()
    backref_attr = getattr(extension, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)


@pytest.mark.parametrize(
    'backref, expected_type',
    [('workflow_element_types', WorkflowElementType)]
)
def test_submission_file_type_model_backref(session_scope_module, backref,
                                            expected_type):
    submission_file_type = (session_scope_module.query(SubmissionFileType)
                                                .first())
    backref_attr = getattr(submission_file_type, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)
