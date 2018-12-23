import os
import shutil
import pytest

# TODO: we temporary use the setup of databoard to create a dataset
from databoard import db
from databoard import deployment_path
from databoard.testing import create_toy_db

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.exceptions import UnknownStateError

from rampdb.model import Submission

from rampdb.tools.api import _setup_db

from rampdb.tools import get_event_nb_folds
from rampdb.tools import get_predictions
from rampdb.tools import get_submission_by_id
from rampdb.tools import get_submission_by_name
from rampdb.tools import get_submission_state
from rampdb.tools import get_submissions
from rampdb.tools import get_time
from rampdb.tools import set_predictions
from rampdb.tools import set_scores
from rampdb.tools import set_submission_error_msg
from rampdb.tools import set_submission_max_ram
from rampdb.tools import set_submission_state
from rampdb.tools import set_time
from rampdb.tools import score_submission

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


def test_hot_test(config_database, db_function):
    print(get_submissions(config_database, 'iris_test', state=None))
    get_submission_by_id(config_database, 7)
    get_submission_by_name(config_database, 'iris_test', 'test_user',
                           'starting_kit_test')
    get_submission_state(config_database, 7)
    get_event_nb_folds(config_database, 'iris_test')
    set_submission_state(config_database, 7, 'trained')
    set_submission_max_ram(config_database, 7, 100)
    set_submission_error_msg(config_database, 7, 'xxxx')


def _change_state_db(config):
    # change the state of one of the submission in the iris event
    db, Session = _setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        sub_id = 1
        sub = session.query(Submission).filter(Submission.id == sub_id).first()
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
    for sub_id, sub_name, sub_path in submissions:
        assert sub_id in expected_id
        assert 'submission_{0:09d}'.format(sub_id) == sub_name
        path_file = os.path.join('submission_{0:09d}'.format(sub_id),
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


@pytest.mark.parametrize("submission_id, state", [(1, 'trained'), (2, 'new')])
def test_submission_state(config_database, db_module, submission_id, state):
    assert get_submission_state(config_database, submission_id) == state


def test_get_event_nb_folds(config_database, db_module):
    assert get_event_nb_folds(config_database, 'iris_test') == 2


def test_set_submission_state(config_database, db_module):
    sub_id = 2
    set_submission_state(config_database, sub_id, 'trained')
    assert get_submission_state(config_database, sub_id) == 'trained'


def test_set_submission_state_unknown_state(config_database, db_module):
    with pytest.raises(UnknownStateError, match='Unrecognized state'):
        set_submission_state(config_database, 2, 'unknown')


def test_check_time(config_database, db_module):
    # check both set_time and get_time function
    sub_id = 1
    path_results = os.path.join(HERE, 'data', 'iris_predictions')
    set_time(config_database, sub_id, path_results)
    print(get_time(config_database, sub_id))


# def test_check_scores(config_database, db_module):
    # check both set_scores and get_scores
#     sub_id = 1
#     path_results = os.path.join(HERE, 'data', 'iris_predictions')
#     set_scores(config_database, sub_id, path_results)
#     submission = get_submission_by_id(config_database, sub_id)
#     cv_fold = submission.on_cv_folds[0]
#     print(cv_fold.scores[0].train_score)


def test_check_predictions(config_database, db_module):
    # check both set_predictions and get_predictions
    sub_id = 1
    path_results = os.path.join(HERE, 'data', 'iris_predictions')
    set_predictions(config_database, sub_id, path_results)
    print(get_predictions(config_database, sub_id))
