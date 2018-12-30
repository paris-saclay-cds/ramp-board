import datetime
import shutil

import pytest

from ramputils import read_config
from ramputils import encode_string
from ramputils.testing import path_config_example

from rampwf.prediction_types.base import BasePrediction
from rampwf.score_types.accuracy import Accuracy

from rampdb.model import EventScoreType
from rampdb.model import EventTeam
from rampdb.model import Model
from rampdb.model import Workflow

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.event import get_event
from rampdb.tools.user import get_team_by_name


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


def test_event_model(session_scope_module):
    event = get_event(session_scope_module, 'iris_test')

    assert repr(event) == 'Event(iris_test)'
    assert issubclass(event.Predictions, BasePrediction)
    assert isinstance(event.workflow, Workflow)
    assert event.workflow.name == 'Classifier'

    event_type_score = event.get_official_score_type(session_scope_module)
    assert event_type_score.name == 'acc'
    assert callable(event.get_official_score_function(session_scope_module))

    assert event.combined_combined_valid_score_str is None
    assert event.combined_combined_test_score_str is None
    assert event.combined_foldwise_valid_score_str is None
    assert event.combined_foldwise_test_score_str is None

    event.combined_combined_valid_score = 0.1
    event.combined_combined_test_score = 0.2
    event.combined_foldwise_valid_score = 0.3
    event.combined_foldwise_test_score = 0.4

    assert (event.get_combined_combined_valid_score_str(
        session_scope_module) == '0.1')
    assert (event.get_combined_combined_test_score_str(
        session_scope_module) == '0.2')
    assert (event.get_combined_foldwise_valid_score_str(
        session_scope_module) == '0.3')
    assert (event.get_combined_foldwise_test_score_str(
        session_scope_module) == '0.4')

    assert event.is_open is True
    # store the original timestamp before to force them
    opening = event.opening_timestamp
    public_opening = event.public_opening_timestamp
    closure = event.closing_timestamp

    event.closing_timestamp = datetime.datetime.utcnow()
    assert event.is_open is False
    assert event.is_closed is True
    event.closing_timestamp = closure
    event.opening_timestamp = (datetime.datetime.utcnow() +
                               datetime.timedelta(days=1))
    assert event.is_open is False
    assert event.is_closed is False
    event.opening_timestamp = opening

    event.closing_timestamp = datetime.datetime.utcnow()
    assert event.is_public_open is False
    assert event.is_closed is True
    event.closing_timestamp = closure
    event.public_opening_timestamp = (datetime.datetime.utcnow() +
                                      datetime.timedelta(days=1))
    assert event.is_public_open is False
    assert event.is_closed is False
    event.public_opening_timestamp = public_opening

    assert event.n_participants == 2

    assert event.n_jobs == 2


def test_event_score_type_model(session_scope_module):
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


def test_event_team(session_scope_module):
    event = get_event(session_scope_module, 'iris_test')
    team = get_team_by_name(session_scope_module, 'test_user')

    event_team = (session_scope_module.query(EventTeam)
                                      .filter(EventTeam.event_id == event.id)
                                      .filter(EventTeam.team_id == team.id)
                                      .one())
    assert repr(event_team) == "Event(iris_test)/Team({})".format(
        encode_string('test_user'))
