import pytest
import shutil

from ramp_database.model import Model
from ramp_database.model import User
from ramp_database.model import Submission
from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template
from ramp_database.testing import create_toy_db
from ramp_database.utils import session_scope
from ramp_database.utils import setup_db
from ramp_database.tools._query import (
    select_submissions_by_state,
    select_submission_by_name,
    select_submission_by_id,
    select_user_by_name,

)


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


def test_select_submissions_by_state(session_scope_module):
    session = session_scope_module

    res = select_submissions_by_state(session, "iris_test", None)
    assert len(res) > 1

    res = select_submissions_by_state(session, "iris_test", "trained")
    assert len(res) == 1

    res = select_submissions_by_state(session, "not_existing", None)
    assert len(res) == 0


def test_select_submissions_by_name(session_scope_module):
    session = session_scope_module

    res = select_submission_by_name(session, "iris_test", "test_user_2",
                                    "starting_kit")
    assert isinstance(res, Submission)

    res = select_submission_by_name(session, "unknown", "unknown", "unknown")
    assert res is None


def test_select_user_by_name(session_scope_module):
    session = session_scope_module

    res = select_user_by_name(session, "test_user_2")
    assert isinstance(res, User)

    res = select_user_by_name(session, "invalid")
    assert res is None


def test_select_submissions_by_id(session_scope_module):
    session = session_scope_module

    res = select_submission_by_id(session, 1)
    assert isinstance(res, Submission)

    res = select_submission_by_id(session, 99999)
    assert res is None
