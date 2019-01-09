import os
import shutil

import pytest

from numpy.testing import assert_array_equal

from ramputils import read_config
from ramputils import generate_ramp_config
from ramputils.utils import import_module_from_source
from ramputils.testing import path_config_example

from rampdb.model import CVFold
from rampdb.model import Event
from rampdb.model import EventScoreType
from rampdb.model import Model
from rampdb.model import Problem
from rampdb.model import Workflow

from rampdb.utils import setup_db
from rampdb.utils import session_scope

from rampdb.testing import create_test_db
from rampdb.testing import create_toy_db
from rampdb.testing import setup_ramp_kits_ramp_data

from rampdb.tools.user import get_user_by_name

from rampdb.tools.event import add_event
from rampdb.tools.event import add_event_admin
from rampdb.tools.event import add_problem
from rampdb.tools.event import add_workflow

from rampdb.tools.event import delete_event
from rampdb.tools.event import delete_problem

from rampdb.tools.event import get_event
from rampdb.tools.event import get_event_admin
from rampdb.tools.event import get_problem
from rampdb.tools.event import get_workflow

from rampdb.tools.event import is_accessible_event
from rampdb.tools.event import is_admin

HERE = os.path.dirname(__file__)


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture
def session_scope_function(config):
    try:
        create_test_db(config)
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


@pytest.fixture(scope='module')
def session_toy_db(config):
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


def test_check_problem(session_scope_function, config):
    problem_names = ['iris', 'boston_housing']
    for problem_name in problem_names:
        setup_ramp_kits_ramp_data(config, problem_name)
        ramp_config = generate_ramp_config(config)
        add_problem(session_scope_function, problem_name,
                    ramp_config['ramp_kits_dir'],
                    ramp_config['ramp_data_dir'])
    problem = get_problem(session_scope_function, problem_names[0])
    assert problem.name == problem_names[0]
    assert isinstance(problem, Problem)
    problem = get_problem(session_scope_function, None)
    assert len(problem) == 2
    assert isinstance(problem, list)

    # Without forcing, we cannot write the same problem twice
    err_msg = 'Attempting to overwrite a problem and delete all linked events'
    with pytest.raises(ValueError, match=err_msg):
        add_problem(session_scope_function, problem_names[0],
                    ramp_config['ramp_kits_dir'], ramp_config['ramp_data_dir'],
                    force=False)

    # Force add the problem
    add_problem(session_scope_function, problem_names[0],
                ramp_config['ramp_kits_dir'], ramp_config['ramp_data_dir'],
                force=True)
    problem = get_problem(session_scope_function, problem_names[0])
    assert problem.name == problem_names[0]
    assert isinstance(problem, Problem)

    delete_problem(session_scope_function, problem_names[0])
    problem = get_problem(session_scope_function, problem_names[0])
    assert problem is None
    problem = get_problem(session_scope_function, None)
    assert len(problem) == 1
    assert isinstance(problem, list)


class WorkflowFaultyElements:
    """Workflow with faulty elements."""
    def __init__(self, case=None):
        self.case = case

    @property
    def element_names(self):
        if self.case == 'multiple-dot':
            return ['too.much.dot.workflow']
        elif self.case == 'unknown-extension':
            return ['function.cpp']


@pytest.mark.parametrize(
    "case, err_msg",
    [('multiple-dot', 'should contain at most one "."'),
     ('unknown-extension', 'Unknown extension')]
)
def test_add_workflow_error(case, err_msg, session_scope_function):
    workflow = WorkflowFaultyElements(case=case)
    with pytest.raises(ValueError, match=err_msg):
        add_workflow(session_scope_function, workflow)
    # TODO: there is no easy way to test a non valid type extension.


def test_check_workflow(session_scope_function):
    # load the workflow from the iris kit which is in the test data
    for kit in ['iris', 'boston_housing']:
        kit_path = os.path.join(HERE, 'data', '{}_kit'.format(kit))
        problem_module = import_module_from_source(
            os.path.join(kit_path, 'problem.py'), 'problem')
        add_workflow(session_scope_function, problem_module.workflow)
    workflow = get_workflow(session_scope_function, None)
    assert len(workflow) == 2
    assert isinstance(workflow, list)
    workflow = get_workflow(session_scope_function, 'Classifier')
    assert workflow.name == 'Classifier'
    assert isinstance(workflow, Workflow)


def _check_event(session, event, event_name, event_title, event_is_public,
                 scores_name):
    assert isinstance(event, Event)
    assert event.name == event_name
    assert event.title == event_title
    assert event.is_public == event_is_public

    score_type = (session.query(EventScoreType)
                         .filter(EventScoreType.event_id == Event.id)
                         .filter(Event.name == event_name)
                         .all())
    for score in score_type:
        assert score.name in scores_name

    # rebuild the fold indices to check if we stored the right one in the
    # database
    cv_folds = (session.query(CVFold)
                       .filter(CVFold.event_id == Event.id)
                       .filter(Event.name == event_name)
                       .all())
    X_train, y_train = event.problem.get_train_data()
    cv = event.problem.module.get_cv(X_train, y_train)
    for ((train_indices, test_indices), stored_fold) in zip(cv, cv_folds):
        assert_array_equal(stored_fold.train_is, train_indices)
        assert_array_equal(stored_fold.test_is, test_indices)


def test_check_event(session_scope_function, config):
    # addition of event require some problem
    problem_names = ['iris', 'boston_housing']
    for problem_name in problem_names:
        setup_ramp_kits_ramp_data(config, problem_name)
        ramp_config = generate_ramp_config(config)
        add_problem(session_scope_function, problem_name,
                    ramp_config['ramp_kits_dir'],
                    ramp_config['ramp_data_dir'])

    for problem_name in problem_names:
        event_name = '{}_test'.format(problem_name)
        event_title = 'event title'
        add_event(session_scope_function, problem_name, event_name,
                  event_title, ramp_config['sandbox_name'],
                  ramp_config['ramp_submissions_dir'],
                  is_public=True, force=False)

    event = get_event(session_scope_function, None)
    assert len(event) == 2
    assert isinstance(event, list)

    event = get_event(session_scope_function, 'iris_test')
    scores_iris = ('acc', 'error', 'nll', 'f1_70')
    _check_event(session_scope_function, event, 'iris_test', 'event title',
                 True, scores_iris)

    # add event for second time without forcing should raise an error
    err_msg = 'Attempting to overwrite existing event.'
    with pytest.raises(ValueError, match=err_msg):
        add_event(session_scope_function, 'iris', 'iris_test', event_title,
                  ramp_config['sandbox_name'],
                  ramp_config['ramp_submissions_dir'], is_public=True,
                  force=False)

    # add event by force
    add_event(session_scope_function, 'iris', 'iris_test', event_title,
              ramp_config['sandbox_name'], ramp_config['ramp_submissions_dir'],
              is_public=True, force=True)
    event = get_event(session_scope_function, 'iris_test')
    _check_event(session_scope_function, event, 'iris_test', 'event title',
                 True, scores_iris)

    delete_event(session_scope_function, 'iris_test')
    event = get_event(session_scope_function, None)
    assert len(event) == 1


def test_check_admin(session_toy_db):
    event_name = 'iris_test'
    user_name = 'test_iris_admin'
    assert is_admin(session_toy_db, event_name, user_name)
    user_name = 'test_user'
    assert not is_admin(session_toy_db, event_name, user_name)
    add_event_admin(session_toy_db, event_name, user_name)
    assert is_admin(session_toy_db, event_name, user_name)
    event_admin = get_event_admin(session_toy_db, event_name, user_name)
    assert event_admin.event.name == event_name
    assert event_admin.admin.name == user_name
    user_name = 'test_user_2'
    assert get_event_admin(session_toy_db, event_name, user_name) is None


@pytest.mark.parametrize(
    "event_name, user_name, is_accessible",
    [('xxx', 'test_iris_admin', False),
     ('iris_test', 'test_user', False),
     ('iris_test', 'test_iris_admin', True),
     ('iris_test', 'test_user_2', True),
     ('boston_housing_test', 'test_user_2', False)]
)
def test_is_accessible_event(session_toy_db, event_name, user_name,
                             is_accessible):
    # force one of the user to not be approved
    if user_name == 'test_user':
        user = get_user_by_name(session_toy_db, user_name)
        user.access_level = 'asked'
        session_toy_db.commit()
    # force an event to be private
    if event_name == 'boston_housing_test':
        event = get_event(session_toy_db, event_name)
        event.is_public = False
        session_toy_db.commit()
    assert is_accessible_event(session_toy_db, event_name,
                               user_name) is is_accessible
