import datetime
import os
import shutil

import pytest
from sqlalchemy.orm import defer

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Event
from ramp_database.model import EventTeam
from ramp_database.model import Model
from ramp_database.model import Submission
from ramp_database.model import SubmissionOnCVFold
from ramp_database.model import User

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope

from ramp_database.testing import add_events
from ramp_database.testing import add_problems
from ramp_database.testing import add_users
from ramp_database.testing import create_test_db

from ramp_database.tools.team import ask_sign_up_team
from ramp_database.tools.team import delete_event_team
from ramp_database.tools.team import sign_up_team


@pytest.fixture
def session_scope_function(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_test_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            add_users(session)
            add_problems(session)
            add_events(session)
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_ask_sign_up_team(session_scope_function):
    event_name, username = 'iris_test', 'test_user'

    ask_sign_up_team(session_scope_function, event_name, username)
    event_team = session_scope_function.query(EventTeam).all()
    assert len(event_team) == 1
    event_team = event_team[0]
    assert event_team.event.name == event_name
    assert event_team.team.name == username
    assert event_team.is_active is True
    assert event_team.last_submission_name is None
    current_datetime = datetime.datetime.now()
    assert event_team.signup_timestamp.year == current_datetime.year
    assert event_team.signup_timestamp.month == current_datetime.month
    assert event_team.signup_timestamp.day == current_datetime.day
    assert event_team.approved is False


def test_sign_up_team(session_scope_function):
    event_name, username = 'iris_test', 'test_user'

    sign_up_team(session_scope_function, event_name, username)
    event_team = session_scope_function.query(EventTeam).all()
    assert len(event_team) == 1
    event_team = event_team[0]

    # when signing up a team, the team is approved and the sandbox is setup:
    # the starting kit is submitted without training it.
    assert event_team.last_submission_name == 'starting_kit'
    assert event_team.approved is True
    # check the status of the sandbox submission
    submission = session_scope_function.query(Submission).all()
    assert len(submission) == 1
    submission = submission[0]
    assert submission.name == 'starting_kit'
    assert submission.event_team == event_team
    submission_file = submission.files[0]
    assert submission_file.name == 'estimator'
    assert submission_file.extension == 'py'
    assert (os.path.join('submission_000000001',
                         'estimator.py') in submission_file.path)
    # check the submission on cv fold
    cv_folds = (session_scope_function.query(SubmissionOnCVFold)
                                      .options(defer("full_train_y_pred"),
                                               defer("test_y_pred"))
                                      .all())
    for fold in cv_folds:
        assert fold.state == 'new'
        assert fold.best is False
        assert fold.contributivity == pytest.approx(0)


def test_delete_event_team(session_scope_function):
    event_name, username = 'iris_test', 'test_user'

    sign_up_team(session_scope_function, event_name, username)
    event_team = session_scope_function.query(EventTeam).all()
    assert len(event_team) == 1

    delete_event_team(session_scope_function, event_name, username)
    event_team = session_scope_function.query(EventTeam).all()
    assert len(event_team) == 0

    # check that the user still exist
    user = (session_scope_function.query(User)
                                  .filter(User.name == username)
                                  .all())
    assert len(user) == 1
    event = (session_scope_function.query(Event)
                                   .filter(Event.name == event_name)
                                   .all())
    assert len(event) == 1
