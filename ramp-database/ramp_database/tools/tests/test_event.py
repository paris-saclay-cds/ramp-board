import os
import shutil

import pytest

from numpy.testing import assert_array_equal

from ramp_utils import read_config
from ramp_utils import generate_ramp_config
from ramp_utils.utils import import_module_from_source
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import CVFold
from ramp_database.model import Event
from ramp_database.model import EventScoreType
from ramp_database.model import Keyword
from ramp_database.model import Model
from ramp_database.model import Problem
from ramp_database.model import ProblemKeyword
from ramp_database.model import Workflow

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope

from ramp_database.testing import create_test_db
from ramp_database.testing import create_toy_db
from ramp_database.testing import setup_ramp_kits_ramp_data

from ramp_database.tools.event import add_event
from ramp_database.tools.event import add_keyword
from ramp_database.tools.event import add_problem
from ramp_database.tools.event import add_problem_keyword
from ramp_database.tools.event import add_workflow

from ramp_database.tools.event import delete_event
from ramp_database.tools.event import delete_problem

from ramp_database.tools.event import get_event
from ramp_database.tools.event import get_keyword_by_name
from ramp_database.tools.event import get_problem
from ramp_database.tools.event import get_problem_keyword_by_name
from ramp_database.tools.event import get_workflow

HERE = os.path.dirname(__file__)


@pytest.fixture
def session_scope_function():
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    try:
        create_test_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(
            ramp_config['ramp']['deployment_dir'], ignore_errors=True
        )
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


@pytest.fixture(scope='module')
def session_toy_db():
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    try:
        create_toy_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(
            ramp_config['ramp']['deployment_dir'], ignore_errors=True
        )
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_check_problem(session_scope_function):
    config = read_config(ramp_config_template())
    problem_names = ['iris', 'boston_housing']
    for problem_name in problem_names:
        # temporary overwrite the configuration to create the problem
        config['ramp']['event'] = problem_name
        config['ramp']['event_name'] = problem_name + '_test'
        config['ramp']['event_title'] = problem_name + ' event'
        config['ramp']['kits_dir'] = os.path.join('ramp-kits', problem_name)
        config['ramp']['data_dir'] = os.path.join('ramp-data', problem_name)
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


def test_check_event(session_scope_function):
    config = read_config(ramp_config_template())
    # addition of event require some problem
    problem_names = ['iris', 'boston_housing']
    for problem_name in problem_names:
        # temporary overwrite the configuration
        config['ramp']['event'] = problem_name
        config['ramp']['event_name'] = problem_name + '_test'
        config['ramp']['event_title'] = problem_name + ' event'
        config['ramp']['kits_dir'] = os.path.join('ramp-kits',
                                                  problem_name)
        config['ramp']['data_dir'] = os.path.join('ramp-data',
                                                  problem_name)
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


def test_check_keyword(session_scope_function):
    add_keyword(session_scope_function, 'keyword', 'data_domain')
    keyword = get_keyword_by_name(session_scope_function, None)
    assert isinstance(keyword, list)
    assert len(keyword) == 1
    keyword = get_keyword_by_name(session_scope_function, 'keyword')
    assert isinstance(keyword, Keyword)
    assert keyword.name == 'keyword'
    assert keyword.type == 'data_domain'
    assert keyword.category is None
    assert keyword.description is None

    err_msg = 'Attempting to update an existing keyword'
    with pytest.raises(ValueError, match=err_msg):
        add_keyword(session_scope_function, 'keyword', 'data_domain')

    add_keyword(session_scope_function, 'keyword', 'data_science_theme',
                category='some cat', description='new description', force=True)
    keyword = get_keyword_by_name(session_scope_function, 'keyword')
    assert keyword.type == 'data_science_theme'
    assert keyword.category == 'some cat'
    assert keyword.description == 'new description'


def test_check_problem_keyword(session_toy_db):
    add_keyword(session_toy_db, 'keyword', 'data_domain')
    add_problem_keyword(session_toy_db, 'iris', 'keyword')
    problem_keyword = get_problem_keyword_by_name(
        session_toy_db, 'iris', 'keyword'
    )
    assert isinstance(problem_keyword, ProblemKeyword)
    assert problem_keyword.problem.name == 'iris'
    assert problem_keyword.keyword.name == 'keyword'
    assert problem_keyword.description is None

    err_msg = 'Attempting to update an existing problem-keyword relationship'
    with pytest.raises(ValueError, match=err_msg):
        add_problem_keyword(session_toy_db, 'iris', 'keyword')

    add_problem_keyword(session_toy_db, 'iris', 'keyword',
                        description='new description', force=True)
    problem_keyword = get_problem_keyword_by_name(
        session_toy_db, 'iris', 'keyword'
    )
    assert problem_keyword.description == 'new description'
