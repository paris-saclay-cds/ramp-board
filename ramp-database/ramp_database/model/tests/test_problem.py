import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from rampwf.prediction_types.base import BasePrediction
from rampwf.workflows import Estimator

from ramp_database.model import Event
from ramp_database.model import Model
from ramp_database.model import ProblemKeyword

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.event import get_problem


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


def test_problem_model(session_scope_module):
    problem = get_problem(session_scope_module, 'iris')

    assert (repr(problem) ==
            "Problem({})\nWorkflow(Estimator)\n\tWorkflow(Estimator): "
            "WorkflowElement(estimator)".format('iris'))

    # check that we can access the problem module and that we have one of the
    # expected function there.
    assert hasattr(problem.module, 'get_train_data')

    assert problem.title == 'Iris classification'
    assert issubclass(problem.Predictions, BasePrediction)
    X_train, y_train = problem.get_train_data()
    assert X_train.shape == (120, 4)
    assert y_train.shape == (120,)
    X_test, y_test = problem.get_test_data()
    assert X_test.shape == (30, 4)
    assert y_test.shape == (30,)
    gt_train = problem.ground_truths_train()
    assert hasattr(gt_train, 'label_names')
    assert gt_train.y_pred.shape == (120, 3)
    gt_test = problem.ground_truths_test()
    assert hasattr(gt_test, 'label_names')
    assert gt_test.y_pred.shape == (30, 3)
    gt_valid = problem.ground_truths_valid([0, 1, 2])
    assert hasattr(gt_valid, 'label_names')
    assert gt_valid.y_pred.shape == (3, 3)

    assert isinstance(problem.workflow_object, Estimator)


@pytest.mark.parametrize(
    'backref, expected_type',
    [('events', Event),
     ('keywords', ProblemKeyword)]
)
def test_problem_model_backref(session_scope_module, backref, expected_type):
    problem = get_problem(session_scope_module, 'iris')
    backref_attr = getattr(problem, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)
