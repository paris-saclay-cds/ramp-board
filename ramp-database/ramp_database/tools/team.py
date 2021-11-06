import logging
import os
from typing import List, Optional

from ..model import EventTeam, Team, UserTeam, Event, User

from .submission import add_submission

from ._query import select_event_by_name
from ._query import select_event_team_by_name
from ._query import select_event_team_by_user_name
from ._query import select_team_by_name
from ._query import select_user_by_name

logger = logging.getLogger("RAMP-DATABASE")


def add_team(
    session, team_name: str, user_name: str, is_individual: bool = True
) -> Team:
    """Create a new team

    Note that the behavior will change depending on whether it's
    an individual team (i.e. team_name == user_name) or not.

    NOTE: before creating a non individual team, you need to leave
    all current teams for the current event with :func:`leave_all_teams`.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The RAMP event name.
    user_name : str
        The name of admin user
    is_individual : bool
        This is an individual team

    Returns
    -------
    team : :class:`ramp_database.model.Team`
        The created team.
    """
    user = select_user_by_name(session, user_name)
    team = Team(name=team_name, admin=user, is_individual=is_individual)
    logger.info(f"Created {team} by {user}")
    session.add(team)
    session.commit()

    if not is_individual:
        user_team = UserTeam(team_id=team.id, user_id=user.id, status="accepted")
        session.add(user_team)

    session.commit()

    return team


def leave_all_teams(session, event_name: str, user_name: str):
    """Leave all teams for a given user and event (except for invididual teams)

    Note that the behavior will change depending on whether it's
    an individual team (i.e. team_name == user_name) or not.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The RAMP event name.
    user_name : str
        The name of admin user
    """
    (
        session.query(UserTeam)
        .filter(
            UserTeam.status == "accepted",
            UserTeam.user_id == User.id,
            User.name == user_name,
            UserTeam.team_id == Team.id,
            EventTeam.team_id == Team.id,
            EventTeam.event_id == Event.id,
            Event.name == event_name,
        )
        .delete(synchronize_session="fetch")
    )
    session.commit()


def ask_sign_up_team(session, event_name, team_name):
    """Register a team to a RAMP event without approving.

    :class:`ramp_database.model.EventTeam` as an attribute ``approved`` set to
    ``False`` by default. Executing this function only create the relationship
    in the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The RAMP event name.
    team_name : str
        The name of the team.

    Returns
    -------
    event : :class:`ramp_database.model.Event`
        The queried Event.
    team : :class:`ramp_database.model.Team`
        The queried team.
    event_team : :class:`ramp_database.model.EventTeam`
        The relationship event-team table.
    """
    event = select_event_by_name(session, event_name)
    team = select_team_by_name(session, team_name)
    event_team = select_event_team_by_name(session, event_name, team_name)
    if event_team is None:
        event_team = EventTeam(event=event, team=team)
        session.add(event_team)
        session.commit()
    return event, team, event_team


def sign_up_team(session, event_name, team_name, user_name: Optional[str] = None):
    """Register a team to a RAMP event and submit the starting kit.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The RAMP event name.
    team_name : str
        The name of the team.
    user_name : str, default=None
        Optional user_name to make the submission as for non individual teams.
        This is only used for audits.
    """
    event, team, event_team = ask_sign_up_team(session, event_name, team_name)
    # setup the sandbox
    path_sandbox_submission = os.path.join(
        event.problem.path_ramp_kit, "submissions", event.ramp_sandbox_name
    )
    submission_name = event.ramp_sandbox_name
    submission = add_submission(
        session,
        event_name,
        team_name,
        submission_name,
        path_sandbox_submission,
        user_name=user_name,
    )
    logger.info("Copying the submission files into the deployment folder")
    logger.info("Adding {}".format(submission))
    event_team.approved = True
    session.commit()


def delete_event_team(session, event_name, team_name):
    """Delete a team from an RAMP event.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The RAMP event name.
    team_name : str
        The name of the team.
    """
    event, team, event_team = ask_sign_up_team(session, event_name, team_name)
    session.delete(event_team)
    session.commit()


def get_event_team_by_name(session, event_name, team_name):
    """Get the event/team given an event and a user.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The RAMP event name.
    team_name : str
        The name of the team.

    Returns
    -------
    event_team : :class:`ramp_database.model.EventTeam`
        The event/team instance queried.
    """
    return select_event_team_by_name(session, event_name, team_name)


def get_event_team_by_user_name(session, event_name, user_name):
    """Get the event/team given an event and a user.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The RAMP event name.
    team_name : str
        The name of the team.

    Returns
    -------
    event_team : :class:`ramp_database.model.EventTeam`
        The event/team instance queried.
    """
    return select_event_team_by_user_name(session, event_name, user_name)


def add_team_member(
    session, team_name: str, user_name: str, status="asked"
) -> List[str]:
    """Add a member to the team

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    team_name : str
        The name of the team.
    user_name : str
        The name of the user.
    status: str
        Membership status.

    Returns
    -------
    errors :
        A list with errors. An empty list if the addition succeded.
    """
    team = select_team_by_name(session, team_name)
    user = select_user_by_name(session, user_name)
    individual_team = select_team_by_name(session, user_name)
    if team.is_individual:
        return [f"Cannot add members to an individual Team({team_name})"]

    event_team = session.query(EventTeam).filter_by(team_id=team.id).one_or_none()
    event = None
    if event_team is not None:
        # Team is signed up to an event. Make sure the user is also signed up
        # to the same event, otherwise they cannot be added to the team
        event = event_team.event
        if (
            session.query(EventTeam)
            .filter_by(event_id=event_team.event.id, team_id=individual_team.id)
            .count()
            == 0
        ):
            return [
                (
                    f"{team} is signed up to {event} however {user} isn't signed up "
                    f"to this event. Therefore cannot invite them."
                )
            ]

    if session.query(UserTeam).filter_by(user=user, team=team).count():
        logging.info(f"add_team_member: {user} is already in {team}. Skipping")

    logging.info(f"Adding {user} to {team} both belonging to {event}")
    user_team = UserTeam(user_id=user.id, team_id=team.id, status=status)
    session.add(user_team)
    session.commit()
    return []


def get_team_members(session, team_name: str, status="accepted") -> List[Team]:
    """Get team members

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    team_name : str
        The name of the team.
    status: str
        With a given status

    Returns
    -------
    teams:
       a list of teams.
    """
    if status not in ["asked", "accepted"]:
        raise ValueError(f"status={status} must be one of ['asked', 'accepted']")
    team = session.query(Team).filter_by(name=team_name).one_or_none()
    if team is None:
        return []
    members = (
        session.query(User)
        .filter(
            User.id == UserTeam.user_id,
            UserTeam.team_id == team.id,
            UserTeam.status == status,
        )
        .distinct()
        .all()
    )
    return list(set(members))


def respond_team_invite(
    session,
    user_name: str,
    team_name: str,
    action: str,
    event_name: Optional[str] = None,
):
    """Respond to a team invite

    Either by accepting or declining. When event_name is not None, the user
    will leave all other non individual teams.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    team_name : str
        The name of the team.
    action: str
        Either accept or decline the invite. One of ['accept', 'decline'].
    event_name : str
        The RAMP event name. Only used to leave all current teams for the event.
        If None, no teams will be left.
    """
    user_team = (
        session.query(UserTeam)
        .filter(
            User.id == UserTeam.user_id,
            Team.id == UserTeam.team_id,
            User.name == user_name,
            Team.name == team_name,
            UserTeam.status == "asked",
        )
        .one_or_none()
    )
    if user_team is None:
        raise ValueError(
            f"Could not find invites for User({user_name}) " f"to Team({team_name})"
        )
    if action == "accept":
        if event_name is not None:
            leave_all_teams(session, event_name, user_name)
        user_team.status = "accepted"
        session.add(user_team)
    elif action == "decline":
        session.delete(user_team)
    else:
        raise ValueError(
            f"unknown action={action} expected one of " f"['accept', 'decline']"
        )
    session.commit()
