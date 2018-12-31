import os
import shutil

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampwf.prediction_types.base import BasePrediction

from rampdb.model import Event
from rampdb.model import EventScoreType
from rampdb.model import Model
from rampdb.model import SubmissionScore
from rampdb.model import Team

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.submission import get_submission_by_id


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


def test_submissions_model(session_scope_module):
    submission = get_submission_by_id(session_scope_module, 5)
    assert (str(submission) ==
            'Submission(iris_test/test_user/starting_kit_test)')
    assert 'Submission(event_name=' in repr(submission)

    assert isinstance(submission.team, Team)
    assert isinstance(submission.event, Event)
    assert submission.official_score_name == 'acc'
    assert isinstance(submission.official_score, SubmissionScore)
    assert all([isinstance(score, EventScoreType)
                for score in submission.score_types])
    assert issubclass(submission.Predictions, BasePrediction)
    assert submission.is_not_sandbox is True
    assert submission.is_error is False
    assert submission.is_public_leaderboard is False
    assert submission.is_private_leaderboard is False
    assert (os.path.join('submissions', 'submission_000000005') in
            submission.path)
    assert submission.basename == 'submission_000000005'
    print(submission.module)
