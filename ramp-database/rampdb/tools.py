from .model import Team
from .model import Event
from .model import EventTeam

__all__ = [
    'get_active_user_event_team',
    'get_n_team_members',
    'get_n_user_teams',
    'get_team_members',
    'get_user_teams',
    'get_user_event_teams',
]


def get_active_user_event_team(event, user):
    # There should always be an active user team, if not, throw an exception
    # The current code works only if each user admins a single team.
    event_team = EventTeam.query.filter_by(
        event=event, team=user.admined_teams[0]).one_or_none()

    return event_team


def get_team_members(team):
    return team.admin


def get_n_team_members(team):
    return len(list(get_team_members(team)))


def get_user_teams(user):
    # This works only if no team mergers. The commented code below
    # is general but slow.
    return Team.query.filter_by(name=user.name).one()


def get_user_event_teams(event_name, user_name):
    # This works only if no team mergers. The commented code below
    # is general but slow.
    event = Event.query.filter_by(name=event_name).one()
    team = Team.query.filter_by(name=user_name).one()

    return EventTeam.query.filter_by(event=event, team=team).one_or_none()


def get_n_user_teams(user):
    return len(get_user_teams(user))
