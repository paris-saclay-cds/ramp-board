import datetime
import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from rampwf.prediction_types.base import BasePrediction
from rampwf.score_types.accuracy import Accuracy

from ramp_database.model.base import set_query_property

from ramp_database.model import CVFold
from ramp_database.model import EventAdmin
from ramp_database.model import EventScoreType
from ramp_database.model import EventTeam
from ramp_database.model import Model
from ramp_database.model import Submission
from ramp_database.model import SubmissionScore
from ramp_database.model import Workflow

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.event import get_event
from ramp_database.tools.user import get_team_by_name


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


def test_event_model_property(session_scope_module):
    event = get_event(session_scope_module, 'iris_test')

    assert repr(event) == 'Event(iris_test)'
    assert issubclass(event.Predictions, BasePrediction)
    assert isinstance(event.workflow, Workflow)
    assert event.workflow.name == 'Estimator'
    assert event.n_participants == 2
    assert event.n_jobs == 2


@pytest.mark.parametrize(
    "opening, public_opening, closure, properties, expected_values",
    [(None, None, None, ['is_open'], [True]),
     (None, None, datetime.datetime.utcnow(), ['is_open', 'is_closed'],
      [False, True]),
     (datetime.datetime.utcnow() + datetime.timedelta(days=1), None, None,
      ['is_open', 'is_closed'], [False, False]),
     (None, None, datetime.datetime.utcnow(), ['is_public_open', 'is_closed'],
      [False, True]),
     (None, datetime.datetime.utcnow() + datetime.timedelta(days=1), None,
      ['is_public_open', 'is_closed'], [False, False])]
)
def test_even_model_timestamp(session_scope_module, opening, public_opening,
                              closure, properties, expected_values):
    # check the property linked to the opening/closure of the event.
    event = get_event(session_scope_module, 'iris_test')

    # store the original timestamp before to force them
    init_opening = event.opening_timestamp
    init_public_opening = event.public_opening_timestamp
    init_closure = event.closing_timestamp

    # set to non-default values the date if necessary
    event.opening_timestamp = opening if opening is not None else init_opening
    event.public_opening_timestamp = (public_opening
                                      if public_opening is not None
                                      else init_public_opening)
    event.closing_timestamp = closure if closure is not None else init_closure

    for prop, exp_val in zip(properties, expected_values):
        assert getattr(event, prop) is exp_val

    # reset the event since that we are sharing the dataset across all the
    # module tests.
    event.opening_timestamp = init_opening
    event.public_opening_timestamp = init_public_opening
    event.closing_timestamp = init_closure


def test_event_model_score(session_scope_module):
    # Make Model usable in declarative mode
    set_query_property(Model, session_scope_module)

    event = get_event(session_scope_module, 'iris_test')

    assert repr(event) == 'Event(iris_test)'
    assert issubclass(event.Predictions, BasePrediction)
    assert isinstance(event.workflow, Workflow)
    assert event.workflow.name == 'Estimator'

    event_type_score = event.official_score_type
    assert event_type_score.name == 'acc'
    event_type_score = event.get_official_score_type(session_scope_module)
    assert event_type_score.name == 'acc'

    assert event.combined_combined_valid_score_str is None
    assert event.combined_combined_test_score_str is None
    assert event.combined_foldwise_valid_score_str is None
    assert event.combined_foldwise_test_score_str is None

    event.combined_combined_valid_score = 0.1
    event.combined_combined_test_score = 0.2
    event.combined_foldwise_valid_score = 0.3
    event.combined_foldwise_test_score = 0.4

    assert event.combined_combined_valid_score_str == '0.1'
    assert event.combined_combined_test_score_str == '0.2'
    assert event.combined_foldwise_valid_score_str == '0.3'
    assert event.combined_foldwise_test_score_str == '0.4'


@pytest.mark.parametrize(
    'backref, expected_type',
    [('score_types', EventScoreType),
     ('event_admins', EventAdmin),
     ('event_teams', EventTeam),
     ('cv_folds', CVFold)]
)
def test_event_model_backref(session_scope_module, backref, expected_type):
    event = get_event(session_scope_module, 'iris_test')
    backref_attr = getattr(event, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)


def test_event_score_type_model_property(session_scope_module):
    event = get_event(session_scope_module, 'iris_test')
    # get only the accuracy score
    event_type_score = \
        (session_scope_module.query(EventScoreType)
                             .filter(EventScoreType.event_id == event.id)
                             .filter(EventScoreType.name == 'acc')
                             .one())

    assert repr(event_type_score) == "acc: Event(iris_test)"
    assert isinstance(event_type_score.score_type_object, Accuracy)
    assert event_type_score.is_lower_the_better is False
    assert event_type_score.minimum == pytest.approx(0)
    assert event_type_score.maximum == pytest.approx(1)
    assert event_type_score.worst == pytest.approx(0)
    assert callable(event_type_score.score_type_object.score_function)


@pytest.mark.parametrize(
    'backref, expected_type',
    [('submissions', SubmissionScore)]
)
def test_event_score_type_model_backref(session_scope_module, backref,
                                        expected_type):
    event = get_event(session_scope_module, 'iris_test')
    # get only the accuracy score
    event_type_score = \
        (session_scope_module.query(EventScoreType)
                             .filter(EventScoreType.event_id == event.id)
                             .filter(EventScoreType.name == 'acc')
                             .one())
    backref_attr = getattr(event_type_score, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)


def test_event_team_model(session_scope_module):
    event = get_event(session_scope_module, 'iris_test')
    team = get_team_by_name(session_scope_module, 'test_user')

    event_team = (session_scope_module.query(EventTeam)
                                      .filter(EventTeam.event_id == event.id)
                                      .filter(EventTeam.team_id == team.id)
                                      .one())
    assert repr(event_team) == "Event(iris_test)/Team({})".format('test_user')


@pytest.mark.parametrize(
    'backref, expected_type',
    [('submissions', Submission)]
)
def test_event_team_model_backref(session_scope_module, backref,
                                  expected_type):
    event = get_event(session_scope_module, 'iris_test')
    team = get_team_by_name(session_scope_module, 'test_user')
    event_team = (session_scope_module.query(EventTeam)
                                      .filter(EventTeam.event_id == event.id)
                                      .filter(EventTeam.team_id == team.id)
                                      .one())
    backref_attr = getattr(event_team, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)
