"""
This module defines different queries which are later used by the API. It
reduces the complexity of having queries and database connection in the same
file. Then, those queries are tested through the public API.
"""
from ..model import Event
from ..model import EventTeam
from ..model import Extension
from ..model import Problem
from ..model import Submission
from ..model import SubmissionFileTypeExtension
from ..model import SubmissionSimilarity
from ..model import Team
from ..model import User
from ..model import Workflow
from ..model import WorkflowElement
from ..model import WorkflowElementType


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
    event : :class:`rampdb.model.Event`
        The queried event.
    """
    return session.query(Event).filter(Event.name == event_name).one_or_none()


def select_user_by_name(session, user_name):
    """Query an user given its name.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    user_name : str
        The username to query.

    Returns
    -------
    user : :class:`rampdb.model.User`
        The queried user.
    """
    if user_name is None:
        return session.query(User).all()
    return session.query(User).filter(User.name == user_name).one()


def select_user_by_email(session, email):
    """Query an user given its email.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    email : str
        The email to query.

    Returns
    -------
    user : :class:`rampdb.model.User`
        The queried user.
    """
    return session.query(User).filter(User.email == email).one_or_none()


def select_team_by_name(session, team_name):
    """Query a team given its name.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    team_name : str
        The team name to query.

    Returns
    -------
    team : :class:`rampdb.model.Team`
        The queried team.
    """
    if team_name is None:
        return session.query(Team).all()
    return session.query(Team).filter(Team.name == team_name).one_or_none()


def select_problem_by_name(session, problem_name):
    """Query a problem given its name.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    problem_name : str
        The problem name to query.

    Returns
    -------
    problem : :class:`rampdb.model.Problem`
        The queried problem.
    """
    if problem_name is None:
        return session.query(Problem).all()
    return (session.query(Problem)
                   .filter(Problem.name == problem_name)
                   .one_or_none())


def select_similarities_by_target(session, target_submission):
    """Query submission similarities given its a target submission.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    target_submission : :class:`rampdb.model.Submission`
        The target submission.

    Returns
    -------
    submission_similarities : list of :class:`rampdb.model.SubmissionSimilarity`
        The queried submission similarity.
    """
    return (session.query(SubmissionSimilarity)
                   .filter(SubmissionSimilarity.target_submission ==
                           target_submission)
                   .all())


def select_similarities_by_source(session, source_submission):
    """Query submission similarities given its a source submission.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    source_submission : :class:`rampdb.model.Submission`
        The source submission.

    Returns
    -------
    submission_similarities : list of :class:`rampdb.model.SubmissionSimilarity`
        The queried submission similarity.
    """
    return (session.query(SubmissionSimilarity)
                   .filter(SubmissionSimilarity.source_submission ==
                           source_submission)
                   .all())


def select_workflow_by_name(session, workflow_name):
    """Query workflow given its name.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    workflow_name : str
        The name of the workflow.

    Returns
    -------
    submission_similarities : :class:`rampdb.model.Workflow`
        The queried workflow.
    """
    if workflow_name is None:
        return session.query(Workflow).all()
    return (session.query(Workflow)
                   .filter(Workflow.name == workflow_name)
                   .one_or_none())


def select_extension_by_name(session, extension_name):
    """Query an extension given its name.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    extension_name : str
        The name of the extension.

    Returns
    -------
    extension : :class:`rampdb.model.Extension`
        The queried extension.
    """
    return (session.query(Extension)
                   .filter(Extension.name == extension_name)
                   .one_or_none())


def select_submission_type_extension_by_extension(session, extension):
    """Query the submission file type extension given its extension.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    extension : :class:`rampdb.model.Extension`
        The extension.

    Returns
    -------
    submission_file_type_extension : :class:`rampdb.model.SubmissionFileTypeExtension`
        The queried submission file type extension.
    """
    return (session.query(SubmissionFileTypeExtension)
                   .filter(SubmissionFileTypeExtension.extension == extension)
                   .one_or_none())


def select_workflow_element_type_by_name(session, workflow_element_type_name):
    """Query the workflow element type given its name.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    workflow_element_type_name : str
        The name of the workflow element type.

    Returns
    -------
    workflow_element_type : :class:`rampdb.model.WorkflowElementType`
        The queried workflow element type.
    """
    return (session.query(WorkflowElementType)
                   .filter(WorkflowElementType.name ==
                           workflow_element_type_name)
                   .one_or_none())


def select_workflow_element_by_workflow_and_type(session, workflow,
                                                 workflow_element_type):
    """Query the workflow element given the workflow and the workflow element
    type.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to query the database.
    workflow : :class:`rampdb.model.Workflow`
        The workflow used to filter.
    workflow_element_type : :class`rampdb.model.WorkflowElement`
        The workflow element type to filter.

    Returns
    -------
    workflow_element : :class:`rampdb.model.WorkflowElement`
        The queried workflow element.
    """
    return (session.query(WorkflowElement)
                   .filter(WorkflowElement.workflow == workflow)
                   .filter(WorkflowElement.workflow_element_type ==
                           workflow_element_type)
                   .one_or_none())
