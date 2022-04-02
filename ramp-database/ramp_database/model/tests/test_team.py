import re
import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Team, UserTeam, User
from ramp_database.model import EventTeam
from ramp_database.model import Model

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.user import get_team_by_name


@pytest.fixture(scope="module")
def session_scope_module(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        with session_scope(database_config["sqlalchemy"]) as session:
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config["sqlalchemy"])
        Model.metadata.drop_all(db)


def test_team_model(session_scope_module):
    team = get_team_by_name(session_scope_module, "test_user")
    expr = r"Team\(name=.*test_user.*, admin_name=.*test_user.*\, is_individual=True\)"
    assert re.match(expr, repr(team))
    assert re.match(r"Team\(.*test_user.*\)", str(team))

    assert team.is_individual is True


def test_user_team_model(session_scope_module):
    """Test user / team association"""
    session = session_scope_module
    # Create a new user and team, so that we can remove them in the end.
    user = User("user-75", "user-75", "", "", "")
    session.add(user)
    session.commit()
    # Create a new non individual team
    team = Team(name="group_team_75", admin=user, is_individual=False)
    session.add(team)
    session.commit()

    # And finally a user-team association
    user_team = UserTeam(team_id=team.id, user_id=user.id, status="accepted")
    session.add(user_team)
    session.commit()

    msg = r"UserTeam\(user_id=4, team_id=4, status='accepted'\)"
    assert re.match(msg, repr(user_team))
    assert team.is_individual is False

    # Check backrefs
    assert user_team.user.name == user.name
    assert user_team.team.name == team.name

    # When a user is deleted, its team and user/team association are also deleted
    session.delete(user)
    session.commit()
    assert session.query(User).filter_by(name="user-75").first() is None
    assert session.query(Team).filter_by(name="group_team_75").first() is None
    assert session.query(UserTeam).filter_by(user_id=4).first() is None


@pytest.mark.parametrize("backref, expected_type", [("team_events", EventTeam)])
def test_event_model_backref(session_scope_module, backref, expected_type):
    team = get_team_by_name(session_scope_module, "test_user")
    backref_attr = getattr(team, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)
