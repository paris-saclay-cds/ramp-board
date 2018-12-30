import datetime
import shutil

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampwf.prediction_types.base import BasePrediction

from rampdb.model import Model
from rampdb.model import Workflow

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.event import get_event


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

    assert event.combined_combined_valid_score_str is None
    assert event.combined_combined_test_score_str is None
    assert event.combined_foldwise_valid_score_str is None
    assert event.combined_foldwise_test_score_str is None

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
