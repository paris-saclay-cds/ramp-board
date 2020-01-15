from ramp_database.model import Submission

from ._query import select_event_admin_by_instance
from ._query import select_event_by_name
from ._query import select_event_team_by_name
from ._query import select_submission_by_name
from ._query import select_user_by_name


def is_admin(session, event_name, user_name):
    """Whether or not a user is administrator or administrate an event.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    user_name : str
        The user name.
    """
    event = select_event_by_name(session, event_name)
    user = select_user_by_name(session, user_name)
    if user.access_level == 'admin':
        return True
    event_admin = select_event_admin_by_instance(session, event, user)
    if event_admin is None:
        return False
    return True


def is_accessible_event(session, event_name, user_name):
    """Whether or not an event is public or and a user is registered to RAMP
    or and admin.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    user_name : str
        The user name.
    """
    event = select_event_by_name(session, event_name)
    user = select_user_by_name(session, user_name)
    if event is None:
        return False
    if user.access_level == 'asked':
        return False
    if event.is_public or is_admin(session, event_name, user_name):
        return True
    return False


def is_accessible_leaderboard(session, event_name, user_name):
    """Whether or not a leaderboard is public or and a user is registered to
    RAMP or and admin.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    user_name : str
        The user name.

    Returns
    -------
    is_accessible : bool
        True if leaderboard can be displayed.
    """
    event = select_event_by_name(session, event_name)
    user = select_user_by_name(session, user_name)
    if not user.is_authenticated or not user.is_active:
        return False
    if is_admin(session, event_name, user_name):
        return True
    if not is_user_signed_up(session, event_name, user_name):
        return False
    if event.is_public_open:
        return True
    return False


def is_accessible_code(session, event_name, user_name,
                       submission_id=None):
    """Whether or not the user can look at the code submission.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The event name.
    user_name : str
        The user name.
    submission_id : int, default=None
        The submission name which you should be shown. Default is the sandbox
        submission.

    Returns
    -------
    is_accessible : bool
        Whether or not the submission can be shown.
    """
    user = select_user_by_name(session, user_name)
    if not user.is_authenticated or not user.is_active:
        return False
    if is_admin(session, event_name, user_name):
        return True
    if not is_user_signed_up(session, event_name, user_name):
        return False

    event = select_event_by_name(session, event_name)
    if event.is_public_open:
        return True
    if submission_id is None:
        submission = select_submission_by_name(
            session, event_name, user_name, event.ramp_sandbox_name
        )
    else:
        submission = (session.query(Submission)
                             .filter_by(id=submission_id)
                             .one_or_none())
    if submission is not None and user == submission.event_team.team.admin:
        return True
    return False


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
    event_team = select_event_team_by_name(session, event_name, user_name)
    if (event_team is not None and
            (event_team.is_active and event_team.approved)):
        return True
    return False


def is_user_sign_up_requested(session, event_name, user_name):
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
    asked : bool
        Whether or not the user had asked to join event or not.
    """
    event_team = select_event_team_by_name(session, event_name, user_name)
    if (event_team is not None and
            (event_team.is_active and not event_team.approved)):
        return True
    return False
