import logging
import os

from ..model import EventTeam

from .submission import add_submission

from ._query import select_event_by_name
from ._query import select_event_team_by_name
from ._query import select_team_by_name

logger = logging.getLogger('RAMP-DATABASE')


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


def sign_up_team(session, event_name, team_name):
    """Register a team to a RAMP event and submit the starting kit.

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
    # setup the sandbox
    path_sandbox_submission = os.path.join(event.problem.path_ramp_kit,
                                           'submissions',
                                           event.ramp_sandbox_name)
    submission_name = event.ramp_sandbox_name
    submission = add_submission(session, event_name, team_name,
                                submission_name, path_sandbox_submission)
    logger.info('Copying the submission files into the deployment folder')
    logger.info('Adding {}'.format(submission))
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


def get_event_team_by_name(session, event_name, user_name):
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
    return select_event_team_by_name(session, event_name, user_name)
