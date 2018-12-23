import os
import shutil

import pytest

import numpy as np
import pandas as pd

from numpy.testing import assert_allclose
from pandas.testing import assert_frame_equal

# TODO: we temporary use the setup of databoard to create a dataset
from databoard import db
from databoard import deployment_path
from databoard.testing import create_toy_db

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.exceptions import UnknownStateError
from rampdb.model import Submission
from rampdb.utils import setup_db

from rampdb.tools.submission import get_event_nb_folds
from rampdb.tools.submission import get_predictions
from rampdb.tools.submission import get_scores
from rampdb.tools.submission import get_submission_by_id
from rampdb.tools.submission import get_submission_by_name
from rampdb.tools.submission import get_submission_state
from rampdb.tools.submission import get_submission_error_msg
from rampdb.tools.submission import get_submission_max_ram
from rampdb.tools.submission import get_submissions
from rampdb.tools.submission import get_time

from rampdb.tools.submission import set_predictions
from rampdb.tools.submission import set_scores
from rampdb.tools.submission import set_submission_error_msg
from rampdb.tools.submission import set_submission_max_ram
from rampdb.tools.submission import set_submission_state
from rampdb.tools.submission import set_time
from rampdb.tools.submission import score_submission

HERE = os.path.dirname(__file__)


@pytest.fixture(scope='module')
def config_database():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture
def db_function():
    try:
        create_toy_db()
        yield
    finally:
        shutil.rmtree(deployment_path, ignore_errors=True)
        db.session.close()
        db.session.remove()
        db.drop_all()


@pytest.fixture(scope='module')
def db_module():
    try:
        create_toy_db()
        _change_state_db(read_config(path_config_example(),
                                     filter_section='sqlalchemy'))
        yield
    finally:
        shutil.rmtree(deployment_path, ignore_errors=True)
        db.session.close()
        db.session.remove()
        db.drop_all()


def _change_state_db(config):
    # change the state of one of the submission in the iris event
    db, Session = setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        submission_id = 1
        sub = (session.query(Submission)
                      .filter(Submission.id == submission_id)
                      .first())
        sub.set_state('trained')
        session.commit()


@pytest.mark.parametrize(
    "state, expected_id",
    [('new', [2, 5, 6, 7, 8, 9, 10]),
     ('trained', [1]),
     ('tested', []),
     (None, [1, 2, 5, 6, 7, 8, 9, 10])]
)
def test_get_submissions(config_database, db_module, state, expected_id):
    submissions = get_submissions(config_database, 'iris_test', state=state)
    assert len(submissions) == len(expected_id)
    for submission_id, sub_name, sub_path in submissions:
        assert submission_id in expected_id
        assert 'submission_{0:09d}'.format(submission_id) == sub_name
        path_file = os.path.join('submission_{0:09d}'.format(submission_id),
                                 'classifier.py')
        assert path_file in sub_path[0]


def test_get_submission_unknown_state(config_database, db_module):
    with pytest.raises(UnknownStateError, match='Unrecognized state'):
        get_submissions(config_database, 'iris_test', state='whatever')


def test_get_submission_by_id(config_database, db_module):
    submission = get_submission_by_id(config_database, 1)
    assert isinstance(submission, Submission)
    assert submission.basename == 'submission_000000001'
    assert os.path.exists(os.path.join(submission.path, 'classifier.py'))
    assert submission.state == 'trained'


def test_get_submission_by_name(config_database, db_module):
    submission = get_submission_by_name(config_database, 'iris_test',
                                        'test_user', 'starting_kit')
    assert isinstance(submission, Submission)
    assert submission.basename == 'submission_000000001'
    assert os.path.exists(os.path.join(submission.path, 'classifier.py'))
    assert submission.state == 'trained'


def test_get_event_nb_folds(config_database, db_module):
    assert get_event_nb_folds(config_database, 'iris_test') == 2


@pytest.mark.parametrize("submission_id, state", [(1, 'trained'), (2, 'new')])
def test_get_submission_state(config_database, db_module, submission_id,
                              state):
    assert get_submission_state(config_database, submission_id) == state


def test_set_submission_state(config_database, db_module):
    submission_id = 2
    set_submission_state(config_database, submission_id, 'trained')
    assert get_submission_state(config_database, submission_id) == 'trained'


def test_set_submission_state_unknown_state(config_database, db_module):
    with pytest.raises(UnknownStateError, match='Unrecognized state'):
        set_submission_state(config_database, 2, 'unknown')


def test_check_time(config_database, db_module):
    # check both set_time and get_time function
    submission_id = 1
    path_results = os.path.join(HERE, 'data', 'iris_predictions')
    set_time(config_database, submission_id, path_results)
    submission_time = get_time(config_database, submission_id)
    expected_df = pd.DataFrame(
        {'fold': [0, 1],
         'train' : [0.032130, 0.002414],
         'valid': [0.000583648681640625, 0.000548362731933594],
         'test': [0.000515460968017578, 0.000481128692626953]}
    ).set_index('fold')
    assert_frame_equal(submission_time, expected_df, check_less_precise=True)


def test_check_scores(config_database, db_module):
    # check both set_scores and get_scores
    submission_id = 1
    path_results = os.path.join(HERE, 'data', 'iris_predictions')
    set_scores(config_database, submission_id, path_results)
    scores = get_scores(config_database, submission_id)
    multi_index = pd.MultiIndex.from_product(
        [[0, 1], ['train', 'valid', 'test']], names=['fold', 'step']
    )
    expected_df = pd.DataFrame(
        {'acc': [0.604167, 0.583333, 0.733333, 0.604167, 0.583333, 0.733333],
         'error': [0.395833, 0.416667, 0.266667, 0.395833, 0.416667, 0.266667],
         'nll': [0.732763, 2.194549, 0.693464, 0.746132, 2.030762, 0.693992],
         'f1_70': [0.333333, 0.33333, 0.666667, 0.33333, 0.33333, 0.666667]},
        index=multi_index
    )
    assert_frame_equal(scores, expected_df, check_less_precise=True)


def test_check_predictions(config_database, db_module):
    # check both set_predictions and get_predictions
    submission_id = 1
    path_results = os.path.join(HERE, 'data', 'iris_predictions')
    set_predictions(config_database, submission_id, path_results)
    predictions = get_predictions(config_database, submission_id)
    for fold_idx in range(2):
        path_fold = os.path.join(path_results, 'fold_{}'.format(fold_idx))
        expected_y_pred_train = np.load(
            os.path.join(path_fold, 'y_pred_train.npz')
        )['y_pred']
        expected_y_pred_test = np.load(
            os.path.join(path_fold, 'y_pred_test.npz')
        )['y_pred']
        assert_allclose(predictions.loc[fold_idx, 'y_pred_train'],
                        expected_y_pred_train)
        assert_allclose(predictions.loc[fold_idx, 'y_pred_test'],
                        expected_y_pred_test)


def test_check_submission_max_ram(config_database, db_module):
    # check both get_submission_max_ram and set_submission_max_ram
    submission_id = 1
    expected_ram = 100.0
    set_submission_max_ram(config_database, submission_id, expected_ram)
    amount_ram = get_submission_max_ram(config_database, submission_id)
    assert amount_ram == pytest.approx(expected_ram)


def test_check_submission_error_msg(config_database, db_module):
    # check both get_submission_error_msg and set_submission_error_msg
    submission_id = 1
    expected_err_msg = 'tag submission as failed'
    set_submission_error_msg(config_database, submission_id, expected_err_msg)
    err_msg = get_submission_error_msg(config_database, submission_id)
    assert err_msg == expected_err_msg


@pytest.mark.filterwarnings('ignore:F-score is ill-defined and being set')
def test_score_submission(config_database, db_module):
    submission_id = 9
    multi_index = pd.MultiIndex.from_product(
        [[0, 1], ['train', 'valid', 'test']], names=['fold', 'step']
    )
    expected_df = pd.DataFrame(
        {'acc': [0.604167, 0.583333, 0.733333, 0.604167, 0.583333, 0.733333],
         'error': [0.395833, 0.416667, 0.266667, 0.395833, 0.416667, 0.266667],
         'nll': [0.732763, 2.194549, 0.693464, 0.746132, 2.030762, 0.693992],
         'f1_70': [0.333333, 0.33333, 0.666667, 0.33333, 0.33333, 0.666667]},
        index=multi_index
    )
    path_results = os.path.join(HERE, 'data', 'iris_predictions')
    with pytest.raises(ValueError, match='Submission state must be "tested"'):
        score_submission(config_database, submission_id)
    set_submission_state(config_database, submission_id, 'tested')
    set_predictions(config_database, submission_id, path_results)
    score_submission(config_database, submission_id)
    scores = get_scores(config_database, submission_id)
    assert_frame_equal(scores, expected_df, check_less_precise=True)
