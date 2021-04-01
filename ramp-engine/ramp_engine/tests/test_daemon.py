import datetime
import os
import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Event
from ramp_database.model import Model
from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_engine.daemon import Daemon


@pytest.fixture
def session_toy(database_connection):
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


def test_daemon_error_init():
    with pytest.raises(ValueError, match="The path xxx is not existing"):
        Daemon(config=database_config_template(), events_dir='xxx')


def test_daemon(session_toy):
    # close both iris events: conda and aws
    event = session_toy.query(Event).filter_by(name="iris_test").one()
    event.closing_timestamp = datetime.datetime.utcnow()
    event = session_toy.query(Event).filter_by(name="iris_aws_test").one()
    event.closing_timestamp = datetime.datetime.utcnow()
    session_toy.commit()

    events_dir = os.path.join(os.path.dirname(__file__), 'events')
    daemon = Daemon(config=database_config_template(), events_dir=events_dir)

    try:
        daemon.launch_dispatchers(session_toy)
        assert len(daemon._proc) == 1
        assert daemon._proc[0][0] == "boston_housing_test"
    finally:
        daemon.kill_dispatcher(None, None)
    assert len(daemon._proc) == 0
