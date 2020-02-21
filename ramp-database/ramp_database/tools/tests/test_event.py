import os
import shutil

import pytest

from numpy.testing import assert_array_equal

from ramp_utils import read_config
from ramp_utils import generate_ramp_config
from ramp_utils.utils import import_module_from_source
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template
from ramp_database.testing import add_users

from ramp_database.model import CVFold
from ramp_database.model import Event
from ramp_database.model import EventScoreType
from ramp_database.model import Keyword
from ramp_database.model import Model
from ramp_database.model import Problem
from ramp_database.model import ProblemKeyword
from ramp_database.model import Submission
from ramp_database.model import Workflow

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope

from ramp_database.testing import create_test_db
from ramp_database.testing import create_toy_db
from ramp_database.testing import ramp_config_boston_housing
from ramp_database.testing import ramp_config_iris
from ramp_database.testing import setup_ramp_kit_ramp_data

from ramp_database.tools.event import add_event
from ramp_database.tools.event import add_event_admin
from ramp_database.tools.event import add_keyword
from ramp_database.tools.event import add_problem
from ramp_database.tools.event import add_problem_keyword
from ramp_database.tools.event import add_workflow

from ramp_database.tools.event import delete_event
from ramp_database.tools.event import delete_problem

from ramp_database.tools.event import get_cv_fold_by_event
from ramp_database.tools.event import get_event
from ramp_database.tools.event import get_event_admin
from ramp_database.tools.event import get_keyword_by_name
from ramp_database.tools.event import get_problem
from ramp_database.tools.event import get_problem_keyword_by_name
from ramp_database.tools.event import get_score_type_by_event
from ramp_database.tools.event import get_workflow

from ramp_database.tools.team import sign_up_team
from ramp_database.tools.team import get_event_team_by_name

HERE = os.path.dirname(__file__)


@pytest.fixture
def session_scope_function(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_test_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            add_users(session)
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


@pytest.fixture(scope='module')
def session_toy_db():
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


def test_check_problem(session_scope_function):
    ramp_configs = {
        'iris': read_config(ramp_config_iris()),
        'boston_housing': read_config(ramp_config_boston_housing())
    }
    for problem_name, ramp_config in ramp_configs.items():
        internal_ramp_config = generate_ramp_config(ramp_config)
        setup_ramp_kit_ramp_data(
            internal_ramp_config, problem_name, mock_html_conversion=True
        )
        add_problem(session_scope_function, problem_name,
                    internal_ramp_config['ramp_kit_dir'],
                    internal_ramp_config['ramp_data_dir'])

    problem_name = 'iris'
    problem = get_problem(session_scope_function, problem_name)
    assert problem.name == problem_name
    assert isinstance(problem, Problem)
    problem = get_problem(session_scope_function, None)
    assert len(problem) == 2
    assert isinstance(problem, list)

    # Without forcing, we cannot write the same problem twice
    internal_ramp_config = generate_ramp_config(ramp_configs[problem_name])
    err_msg = 'Attempting to overwrite a problem and delete all linked events'
    with pytest.raises(ValueError, match=err_msg):
        add_problem(
            session_scope_function, problem_name,
            internal_ramp_config['ramp_kit_dir'],
            internal_ramp_config['ramp_data_dir'],
            force=False
        )

    # Force add the problem
    add_problem(
        session_scope_function, problem_name,
        internal_ramp_config['ramp_kit_dir'],
        internal_ramp_config['ramp_data_dir'],
        force=True
    )
    problem = get_problem(session_scope_function, problem_name)
    assert problem.name == problem_name
    assert isinstance(problem, Problem)

    delete_problem(session_scope_function, problem_name)
    problem = get_problem(session_scope_function, problem_name)
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
        if self.case == 'unknown-extension':
            return ['function.cpp']


@pytest.mark.parametrize(
    "case, err_msg",
    [('unknown-extension', 'Unknown extension')]
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
    workflow = get_workflow(session_scope_function, 'Estimator')
    assert workflow.name == 'Estimator'
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
    ramp_configs = {
        'iris': read_config(ramp_config_iris()),
        'boston_housing': read_config(ramp_config_boston_housing())
    }
    for problem_name, ramp_config in ramp_configs.items():
        internal_ramp_config = generate_ramp_config(ramp_config)
        setup_ramp_kit_ramp_data(
            internal_ramp_config, problem_name, depth=1,
            mock_html_conversion=True
        )
        add_problem(session_scope_function, problem_name,
                    internal_ramp_config['ramp_kit_dir'],
                    internal_ramp_config['ramp_data_dir'])

    for problem_name, ramp_config in ramp_configs.items():
        internal_ramp_config = generate_ramp_config(ramp_config)
        add_event(session_scope_function, problem_name,
                  internal_ramp_config['event_name'],
                  internal_ramp_config['event_title'],
                  internal_ramp_config['sandbox_name'],
                  internal_ramp_config['ramp_submissions_dir'],
                  is_public=True, force=False)

    event = get_event(session_scope_function, None)
    assert len(event) == 2
    assert isinstance(event, list)

    problem_name = 'iris'
    internal_ramp_config = generate_ramp_config(ramp_configs[problem_name])
    event = get_event(
        session_scope_function, internal_ramp_config['event_name']
    )
    scores_iris = ('acc', 'error', 'nll', 'f1_70')
    _check_event(
        session_scope_function, event, internal_ramp_config['event_name'],
        internal_ramp_config['event_title'], True, scores_iris
    )

    # add event for second time without forcing should raise an error
    err_msg = 'Attempting to overwrite existing event.'
    with pytest.raises(ValueError, match=err_msg):
        add_event(
            session_scope_function, problem_name,
            internal_ramp_config['event_name'],
            internal_ramp_config['event_title'],
            internal_ramp_config['sandbox_name'],
            internal_ramp_config['ramp_submissions_dir'],
            is_public=True,
            force=False
        )

    # add event by force
    add_event(
        session_scope_function, problem_name,
        internal_ramp_config['event_name'],
        internal_ramp_config['event_title'],
        internal_ramp_config['sandbox_name'],
        internal_ramp_config['ramp_submissions_dir'],
        is_public=True, force=True
    )
    event = get_event(
        session_scope_function, internal_ramp_config['event_name']
    )
    _check_event(
        session_scope_function, event, internal_ramp_config['event_name'],
        internal_ramp_config['event_title'], True, scores_iris
    )

    delete_event(session_scope_function, internal_ramp_config['event_name'])
    event = get_event(session_scope_function, None)
    assert len(event) == 1

    # try to add an event that does not start with the problem name
    err_msg = "The event name should start with the problem name: 'iris_'"
    with pytest.raises(ValueError, match=err_msg):
        add_event(
            session_scope_function, problem_name,
            "xxxx",
            internal_ramp_config['event_title'],
            internal_ramp_config['sandbox_name'],
            internal_ramp_config['ramp_submissions_dir'],
            is_public=True,
            force=False
        )


def test_delete_event(session_scope_function):
    ###########################################################################
    # Setup the problem/event

    # add sample problem
    problem_name = 'iris'
    ramp_config = read_config(ramp_config_iris())

    internal_ramp_config = generate_ramp_config(ramp_config)
    setup_ramp_kit_ramp_data(internal_ramp_config, problem_name, depth=1)
    add_problem(session_scope_function, problem_name,
                internal_ramp_config['ramp_kit_dir'],
                internal_ramp_config['ramp_data_dir'])

    event_name = 'iris_test_delete'
    # add sample event
    add_event(session_scope_function, problem_name,
              event_name,
              internal_ramp_config['event_title'],
              internal_ramp_config['sandbox_name'],
              internal_ramp_config['ramp_submissions_dir'],
              is_public=True, force=False)

    event = get_event(session_scope_function,
                      event_name)
    assert event

    # add event-team
    username = 'test_user'
    sign_up_team(session_scope_function, event_name, username)
    event_team = get_event_team_by_name(session_scope_function,
                                        event_name, username)
    assert event_team

    # add event admin
    add_event_admin(session_scope_function, event_name, username)
    assert get_event_admin(session_scope_function, event_name, username)

    # check if the event is connected to any score type in the database
    event_score_types = get_score_type_by_event(session_scope_function, event)
    assert len(event_score_types) > 0

    # check if the event is connected to any cv_fold in the database
    event_cv_fold = get_cv_fold_by_event(session_scope_function, event)
    assert len(event_cv_fold) > 0

    # add the submission to the event
    from ramp_database.tools.submission import get_submission_by_id
    submission = get_submission_by_id(session_scope_function, event.id)
    assert submission

    # ensure that the changes have been committed in the database
    session_scope_function.commit()

    ###########################################################################
    # Check the behaviour of delete_event

    delete_event(session_scope_function, event_name)
    # make sure event and all the connections were deleted
    event_test = get_event(session_scope_function, None)
    assert len(event_test) == 0
    assert not get_event_team_by_name(session_scope_function,
                                      event_name, username)
    assert not get_event_admin(session_scope_function, event_name, username)
    event_score_types = get_score_type_by_event(session_scope_function, event)
    assert len(event_score_types) == 0
    event_cv_fold = get_cv_fold_by_event(session_scope_function, event)
    assert not event_cv_fold
    assert len(session_scope_function.query(Submission).all()) == 0


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
