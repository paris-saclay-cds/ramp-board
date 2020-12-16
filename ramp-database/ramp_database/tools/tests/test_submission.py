import datetime
import os
import shutil

import pytest

import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
from pandas.testing import assert_frame_equal

from ramp_utils import read_config
from ramp_utils import generate_ramp_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.exceptions import DuplicateSubmissionError
from ramp_database.exceptions import MissingSubmissionFileError
from ramp_database.exceptions import MissingExtensionError
from ramp_database.exceptions import TooEarlySubmissionError
from ramp_database.exceptions import UnknownStateError

from ramp_database.model import Event
from ramp_database.model import Model
from ramp_database.model import Submission
from ramp_database.model import SubmissionSimilarity
from ramp_database.testing import add_events
from ramp_database.testing import add_problems
from ramp_database.testing import add_users
from ramp_database.testing import create_toy_db
from ramp_database.testing import create_test_db
from ramp_database.testing import ramp_config_iris
from ramp_database.testing import sign_up_teams_to_events
from ramp_database.utils import setup_db
from ramp_database.utils import session_scope

from ramp_database.tools.user import add_user_interaction
from ramp_database.tools.user import get_user_by_name

from ramp_database.tools.submission import add_submission
from ramp_database.tools.submission import add_submission_similarity

from ramp_database.tools.submission import get_bagged_scores
from ramp_database.tools.submission import get_event_nb_folds
from ramp_database.tools.submission import get_predictions
from ramp_database.tools.submission import get_scores
from ramp_database.tools.submission import get_source_submissions
from ramp_database.tools.submission import get_submission_by_id
from ramp_database.tools.submission import get_submission_by_name
from ramp_database.tools.submission import get_submission_state
from ramp_database.tools.submission import get_submission_error_msg
from ramp_database.tools.submission import get_submission_max_ram
from ramp_database.tools.submission import get_submissions
from ramp_database.tools.submission import get_time

from ramp_database.tools.submission import set_bagged_scores
from ramp_database.tools.submission import set_predictions
from ramp_database.tools.submission import set_scores
from ramp_database.tools.submission import set_submission_error_msg
from ramp_database.tools.submission import set_submission_max_ram
from ramp_database.tools.submission import set_submission_state
from ramp_database.tools.submission import set_time

from ramp_database.tools.submission import score_submission
from ramp_database.tools.submission import submit_starting_kits

HERE = os.path.dirname(__file__)
ID_SUBMISSION = 7


@pytest.fixture
def base_db(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_test_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def _change_state_db(session):
    # change the state of one of the submission in the iris event
    submission_id = 1
    sub = (session.query(Submission)
                  .filter(Submission.id == submission_id)
                  .first())
    sub.set_state('trained')
    session.commit()


@pytest.fixture(scope='module')
def session_scope_module(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            _change_state_db(session)
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def _setup_sign_up(session):
    # asking to sign up required a user, a problem, and an event.
    add_users(session)
    add_problems(session)
    add_events(session)
    sign_up_teams_to_events(session)
    return 'iris_test', 'test_user'


def test_add_submission_create_new_submission(base_db):
    # check that we can make a new submission to the database
    # it will require to have already a team and an event
    session = base_db
    config = ramp_config_template()
    event_name, username = _setup_sign_up(session)
    ramp_config = generate_ramp_config(read_config(config))

    submission_name = 'random_forest_10_10'
    path_submission = os.path.join(
        os.path.dirname(ramp_config['ramp_sandbox_dir']), submission_name
    )
    add_submission(session, event_name, username, submission_name,
                   path_submission)
    all_submissions = get_submissions(session, event_name, None)
    # check that the submissions have been copied
    for sub_id, _, _ in all_submissions:
        sub = get_submission_by_id(session, sub_id)
        assert os.path.exists(sub.path)
        assert os.path.exists(os.path.join(sub.path, 'estimator.py'))

    # `sign_up_team` make a submission (sandbox) by user. This submission will
    # be the third submission.
    assert len(all_submissions) == 3
    # check that the number of submissions for an event was updated
    event = session.query(Event).filter(Event.name == event_name).one_or_none()
    assert event.n_submissions == 1
    submission = get_submission_by_name(session, event_name, username,
                                        submission_name)
    assert submission.name == submission_name
    submission_file = submission.files[0]
    assert submission_file.name == 'estimator'
    assert submission_file.extension == 'py'
    assert (os.path.join('submission_00000000' + str(ID_SUBMISSION),
                         'estimator.py') in submission_file.path)


def test_add_submission_too_early_submission(base_db):
    # check that we raise an error when the elapsed time was not large enough
    # between the new submission and the previous submission
    session = base_db
    config = ramp_config_template()
    event_name, username = _setup_sign_up(session)
    ramp_config = generate_ramp_config(read_config(config))

    # check that we have an awaiting time for the event
    event = (session.query(Event)
                    .filter(Event.name == event_name)
                    .one_or_none())
    assert event.min_duration_between_submissions == 900

    # make 2 submissions which are too close from each other
    for submission_idx, submission_name in enumerate(['random_forest_10_10',
                                                      'too_early_submission']):
        path_submission = os.path.join(
            os.path.dirname(ramp_config['ramp_sandbox_dir']), submission_name
        )
        if submission_idx == 1:
            err_msg = 'You need to wait'
            with pytest.raises(TooEarlySubmissionError, match=err_msg):
                add_submission(session, event_name, username, submission_name,
                               path_submission)
        else:
            add_submission(session, event_name, username, submission_name,
                           path_submission)


def test_make_submission_resubmission(base_db):
    # check that resubmitting the a submission with the same name will raise
    # an error
    session = base_db
    config = ramp_config_template()
    event_name, username = _setup_sign_up(session)
    ramp_config = generate_ramp_config(read_config(config))

    # submitting the starting_kit which is used as the default submission for
    # the sandbox should raise an error
    err_msg = ('Submission "starting_kit" of team "test_user" at event '
               '"iris_test" exists already')
    with pytest.raises(DuplicateSubmissionError, match=err_msg):
        add_submission(session, event_name, username,
                       os.path.basename(ramp_config['ramp_sandbox_dir']),
                       ramp_config['ramp_sandbox_dir'])

    # submitting twice a normal submission should raise an error as well
    submission_name = 'random_forest_10_10'
    path_submission = os.path.join(
        os.path.dirname(ramp_config['ramp_sandbox_dir']), submission_name
    )
    # first submission
    add_submission(session, event_name, username, submission_name,
                   path_submission,)
    # mock that we scored the submission
    set_submission_state(session, ID_SUBMISSION, 'scored')
    # second submission
    err_msg = ('Submission "random_forest_10_10" of team "test_user" at event '
               '"iris_test" exists already')
    with pytest.raises(DuplicateSubmissionError, match=err_msg):
        add_submission(session, event_name, username, submission_name,
                       path_submission)

    # a resubmission can take place if it is tagged as "new" or failed

    # mock that the submission failed during the training
    set_submission_state(session, ID_SUBMISSION, 'training_error')
    add_submission(session, event_name, username, submission_name,
                   path_submission)
    # mock that the submissions are new submissions
    set_submission_state(session, ID_SUBMISSION, 'new')
    add_submission(session, event_name, username, submission_name,
                   path_submission)


def test_add_submission_wrong_submission_files(base_db):
    # check that we raise an error if the file required by the workflow is not
    # present in the submission or that it has the wrong extension
    session = base_db
    config = ramp_config_template()
    event_name, username = _setup_sign_up(session)
    ramp_config = generate_ramp_config(read_config(config))

    submission_name = 'corrupted_submission'
    path_submission = os.path.join(
        os.path.dirname(ramp_config['ramp_sandbox_dir']), submission_name
    )
    os.makedirs(path_submission)

    # case that there is not files in the submission
    err_msg = 'No file corresponding to the workflow element'
    with pytest.raises(MissingSubmissionFileError, match=err_msg):
        add_submission(session, event_name, username, submission_name,
                       path_submission)

    # case that there is not file corresponding to the workflow component
    filename = os.path.join(path_submission, 'unknown_file.xxx')
    open(filename, "w+").close()
    err_msg = 'No file corresponding to the workflow element'
    with pytest.raises(MissingSubmissionFileError, match=err_msg):
        add_submission(session, event_name, username, submission_name,
                       path_submission)

    # case that we have the correct filename but not the right extension
    filename = os.path.join(path_submission, 'estimator.xxx')
    open(filename, "w+").close()
    err_msg = 'All extensions "xxx" are unknown for the submission'
    with pytest.raises(MissingExtensionError, match=err_msg):
        add_submission(session, event_name, username, submission_name,
                       path_submission)


def test_submit_starting_kits(base_db):
    session = base_db
    config = ramp_config_iris()
    event_name, username = _setup_sign_up(session)
    ramp_config = generate_ramp_config(read_config(config))

    submit_starting_kits(session, event_name, username,
                         ramp_config['ramp_kit_submissions_dir'])

    submissions = get_submissions(session, event_name, None)
    submissions_id = [sub[0] for sub in submissions]
    assert len(submissions) == 5
    expected_submission_name = {'starting_kit', 'starting_kit_test',
                                'random_forest_10_10', 'error'}
    submission_name = {get_submission_by_id(session, sub_id).name
                       for sub_id in submissions_id}
    assert submission_name == expected_submission_name


@pytest.mark.parametrize(
    "state, expected_id",
    [('new', [2, 7, 8, 9, 10, 11, 12]),
     ('trained', [1]),
     ('tested', []),
     (None, [1, 2, 7, 8, 9, 10, 11, 12])]
)
def test_get_submissions(session_scope_module, state, expected_id):
    submissions = get_submissions(session_scope_module, 'iris_test',
                                  state=state)
    assert len(submissions) == len(expected_id)
    for submission_id, sub_name, sub_path in submissions:
        assert submission_id in expected_id
        assert 'submission_{:09d}'.format(submission_id) == sub_name
        path_file = os.path.join('submission_{:09d}'.format(submission_id),
                                 'estimator.py')
        assert path_file in sub_path[0]


def test_get_submission_unknown_state(session_scope_module):
    with pytest.raises(UnknownStateError, match='Unrecognized state'):
        get_submissions(session_scope_module, 'iris_test', state='whatever')


def test_get_submission_by_id(session_scope_module):
    submission = get_submission_by_id(session_scope_module, 1)
    assert isinstance(submission, Submission)
    assert submission.basename == 'submission_000000001'
    assert os.path.exists(os.path.join(submission.path, 'estimator.py'))
    assert submission.state == 'trained'


def test_get_submission_by_name(session_scope_module):
    submission = get_submission_by_name(session_scope_module, 'iris_test',
                                        'test_user', 'starting_kit')
    assert isinstance(submission, Submission)
    assert submission.basename == 'submission_000000001'
    assert os.path.exists(os.path.join(submission.path, 'estimator.py'))
    assert submission.state == 'trained'


def test_get_event_nb_folds(session_scope_module):
    assert get_event_nb_folds(session_scope_module, 'iris_test') == 2


@pytest.mark.parametrize("submission_id, state", [(1, 'trained'), (2, 'new')])
def test_get_submission_state(session_scope_module, submission_id, state):
    assert get_submission_state(session_scope_module, submission_id) == state


def test_set_submission_state(session_scope_module):
    submission_id = 2
    set_submission_state(session_scope_module, submission_id, 'trained')
    state = get_submission_state(session_scope_module, submission_id)
    assert state == 'trained'


def test_set_submission_state_unknown_state(session_scope_module):
    with pytest.raises(UnknownStateError, match='Unrecognized state'):
        set_submission_state(session_scope_module, 2, 'unknown')


def test_check_time(session_scope_module):
    # check both set_time and get_time function
    submission_id = 1
    path_results = os.path.join(HERE, 'data', 'iris_predictions')
    set_time(session_scope_module, submission_id, path_results)
    submission_time = get_time(session_scope_module, submission_id)
    expected_df = pd.DataFrame(
        {'fold': [0, 1],
         'train': [0.032130, 0.002414],
         'valid': [0.000583648681640625, 0.000548362731933594],
         'test': [0.000515460968017578, 0.000481128692626953]}
    ).set_index('fold')
    assert_frame_equal(submission_time, expected_df, check_less_precise=True)


def test_check_scores(session_scope_module):
    # check both set_scores and get_scores
    submission_id = 1
    path_results = os.path.join(HERE, 'data', 'iris_predictions')
    set_scores(session_scope_module, submission_id, path_results)
    scores = get_scores(session_scope_module, submission_id)
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


def test_check_bagged_scores(session_scope_module):
    # check both set_bagged_scores and get_bagged_scores
    submission_id = 1
    path_results = os.path.join(HERE, 'data', 'iris_predictions')
    set_bagged_scores(session_scope_module, submission_id, path_results)
    scores = get_bagged_scores(session_scope_module, submission_id)
    multi_index = pd.MultiIndex(levels=[['test', 'valid'], [0, 1]],
                                codes=[[0, 0, 1, 1], [0, 1, 0, 1]],
                                names=['step', 'n_bag'])
    expected_df = pd.DataFrame(
        {'acc': [0.70833333333, 0.70833333333, 0.65, 0.6486486486486],
         'error': [0.29166666667, 0.29166666667, 0.35, 0.35135135135],
         'nll': [0.80029268745, 0.66183018275, 0.52166532641, 0.58510855181],
         'f1_70': [0.66666666667, 0.33333333333, 0.33333333333,
                   0.33333333333]},
        index=multi_index
    )
    expected_df.columns = expected_df.columns.rename('scores')
    assert_frame_equal(scores, expected_df, check_less_precise=True)


def test_check_predictions(session_scope_module):
    # check both set_predictions and get_predictions
    submission_id = 1
    path_results = os.path.join(HERE, 'data', 'iris_predictions')
    set_predictions(session_scope_module, submission_id, path_results)
    predictions = get_predictions(session_scope_module, submission_id)
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


def test_check_submission_max_ram(session_scope_module):
    # check both get_submission_max_ram and set_submission_max_ram
    submission_id = 1
    expected_ram = 100.0
    set_submission_max_ram(session_scope_module, submission_id, expected_ram)
    amount_ram = get_submission_max_ram(session_scope_module, submission_id)
    assert amount_ram == pytest.approx(expected_ram)


def test_check_submission_error_msg(session_scope_module):
    # check both get_submission_error_msg and set_submission_error_msg
    submission_id = 1
    expected_err_msg = 'tag submission as failed'
    set_submission_error_msg(session_scope_module, submission_id,
                             expected_err_msg)
    err_msg = get_submission_error_msg(session_scope_module, submission_id)
    assert err_msg == expected_err_msg


@pytest.mark.filterwarnings('ignore:F-score is ill-defined and being set')
def test_score_submission(session_scope_module):
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
        score_submission(session_scope_module, submission_id)
    set_submission_state(session_scope_module, submission_id, 'tested')
    set_predictions(session_scope_module, submission_id, path_results)
    score_submission(session_scope_module, submission_id)
    scores = get_scores(session_scope_module, submission_id)
    assert_frame_equal(scores, expected_df, check_less_precise=True)


def test_get_source_submission(session_scope_module):
    # since we do not record interaction without the front-end, we should get
    # an empty list
    submission_id = 1
    submissions = get_source_submissions(session_scope_module, submission_id)
    assert not submissions
    # we simulate some user interaction
    # case 1: the interaction come after the file to be submitted so there
    # is no submission to show
    submission = get_submission_by_id(session_scope_module, submission_id)
    event = submission.event_team.event
    user = submission.event_team.team.admin
    add_user_interaction(
        session_scope_module, user=user, interaction='looking at submission',
        event=event, submission=get_submission_by_id(session_scope_module, 2)
    )
    submissions = get_source_submissions(session_scope_module, submission_id)
    assert not submissions
    # case 2: we postpone the time of the submission to simulate that we
    # already check other submission.
    submission.submission_timestamp += datetime.timedelta(days=1)
    submissions = get_source_submissions(session_scope_module, submission_id)
    assert submissions
    assert all([sub.event_team.event.name == event.name
                for sub in submissions])


def test_add_submission_similarity(session_scope_module):
    user = get_user_by_name(session_scope_module, 'test_user')
    source_submission = get_submission_by_id(session_scope_module, 1)
    target_submission = get_submission_by_id(session_scope_module, 2)
    add_submission_similarity(session_scope_module, 'target_credit', user,
                              source_submission, target_submission, 0.5,
                              datetime.datetime.utcnow())
    similarity = session_scope_module.query(SubmissionSimilarity).all()
    assert len(similarity) == 1
    similarity = similarity[0]
    assert similarity.type == 'target_credit'
    assert similarity.user == user
    assert similarity.source_submission == source_submission
    assert similarity.target_submission == target_submission
    assert similarity.similarity == pytest.approx(0.5)
    assert isinstance(similarity.timestamp, datetime.datetime)
