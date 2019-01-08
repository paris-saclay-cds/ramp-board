import logging
import os
import shutil

from ..model import EventTeam

from .submission import add_submission

from ._query import select_event_by_name
from ._query import select_event_team_by_name
from ._query import select_team_by_name

logger = logging.getLogger('DATABASE')


def ask_sign_up_team(session, event_name, team_name):
    """Register a team to a RAMP event without approving.

    :class:`rampdb.model.EventTeam` as an attribute ``approved`` set to
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
    event : :class:`rampdb.model.Event`
        The queried Event.
    team : :class:`rampdb.model.Team`
        The queried team.
    event_team : :class:`rampdb.model.EventTeam`
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


def sign_up_team(session, event_name, team_name, path_sandbox_submission,
                 path_ramp_submissions):
    """Register a team to a RAMP event and submit the starting kit.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The RAMP event name.
    team_name : str
        The name of the team.
    path_sandbox_submission : str
        Path to the sandbox submission. It will corresponds to the key
        `ramp_sandbox_dir` of the dictionary created with
        :func:`ramputils.generate_ramp_config`.
    path_ramp_submissions : str
        Path to the deployment RAMP submissions directory. It will corresponds
        to the key `ramp_submissions_dir` of the dictionary created with
        :func:`ramputils.generate_ramp_config`.
    """
    event, team, event_team = ask_sign_up_team(session, event_name, team_name)
    # setup the sandbox
    submission_name = os.path.basename(path_sandbox_submission)
    submission = add_submission(session, event_name, team_name,
                                submission_name, path_sandbox_submission,
                                path_ramp_submissions, True)
    if os.path.exists(submission.path):
        shutil.rmtree(submission.path)
    os.makedirs(submission.path)
    for filename in submission.f_names:
        shutil.copy2(src=os.path.join(path_sandbox_submission, filename),
                     dst=os.path.join(submission.path, filename))
    logger.info('Copying the submission files into the deployment folder')
    logger.info('Adding {}'.format(submission))
    # TODO: be sure that we send an email
    # for user in get_team_members(team):
    #     send_mail(to=user.email,
    #               subject='signed up for {} as team {}'.format(
    #                   event_name, team_name),
    #               body='')
    event_team.approved = True
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
    event_team : :class:`rampdb.model.EventTeam`
        The event/team instance queried.
    """
    return select_event_team_by_name(session, event_name, user_name)


def is_user_signed_up(session, event_name, user_name):
    """Whether or not user signed up to an event.

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
    is_signed_up : bool
        Whether or not the user is signed up for the event.
    """
    event_team = get_event_team_by_name(session, event_name, user_name)
    if (event_team is not None and
            (event_team.is_active and event_team.approved)):
        return True
    return False
