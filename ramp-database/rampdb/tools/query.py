"""
This module defines different queries which are later used by the API. It
reduces the complexity of having queries and database connection in the same
file. Then, those queries are tested through the public API.
"""
from ..model import Event
from ..model import EventTeam
from ..model import Team
from ..model import Submission


def select_submissions_by_state(session, event_name, state):
    """Query all submissions for a given event with given state.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    event_name : str
        The name of the RAMP event.
    state : str
        The state of the submissions to query.

    Returns
    -------
    submissions : list of :class:`rampdb.model.Submission`
        The queried list of submissions.
    """
    q = (session.query(Submission)
                   .filter(Event.name == event_name)
                   .filter(Event.id == EventTeam.event_id)
                   .filter(EventTeam.id == Submission.event_team_id)
                   .order_by(Submission.submission_timestamp))
    if state is None:
        return q.all()
    return q.filter(Submission.state == state).all()


def select_submission_by_id(session, submission_id):
    """Query a submission given its id.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    submission_id : int
        The identification of the submission.

    Returns
    -------
    submission : :class:`rampdb.model.Submission`
        The queried submission.
    """
    return (session.query(Submission)
                   .filter(Submission.id == submission_id)
                   .first())


def select_submission_by_name(session, event_name, team_name, name):
    """Query a submission given the event, team, and submission names.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    event_name : str
        The name of the RAMP event.
    team_name : str
        The name of the team.
    name : str
        The submission name.

    Returns
    -------
    submission : :class:`rampdb.model.Submission`
        The queried submission.
    """
    return (session.query(Submission)
                   .filter(Event.name == event_name)
                   .filter(Event.id == EventTeam.event_id)
                   .filter(Team.name == team_name)
                   .filter(Team.id == EventTeam.team_id)
                   .filter(EventTeam.id == Submission.event_team_id)
                   .filter(Submission.name == name)
                   .order_by(Submission.submission_timestamp)
                   .one())


def select_event_by_name(session, event_name):
    """Query an event given its name.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    event_name : str
        The name of the RAMP event.

    Returns
    -------
    event : :rampdb.model.Event`
        The queried event.
    """
    return session.query(Event).filter(Event.name == event_name).one()
