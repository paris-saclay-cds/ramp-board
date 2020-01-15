import logging
import os

from sqlalchemy.orm.exc import NoResultFound

from ramp_utils.utils import import_module_from_source

from ._query import select_event_admin_by_instance
from ._query import select_event_by_name
from ._query import select_extension_by_name
from ._query import select_problem_by_name
from ._query import select_submissions_by_state
from ._query import select_similarities_by_source
from ._query import select_similarities_by_target
from ._query import select_submission_by_id
from ._query import select_submission_type_extension_by_extension
from ._query import select_user_by_name
from ._query import select_workflow_by_name
from ._query import select_workflow_element_by_workflow_and_type
from ._query import select_workflow_element_type_by_name

from ..model import CVFold
from ..model import Event
from ..model import EventAdmin
from ..model import EventScoreType
from ..model import Keyword
from ..model import Problem
from ..model import ProblemKeyword
from ..model import Workflow
from ..model import WorkflowElement
from ..model import WorkflowElementType

logger = logging.getLogger('RAMP-DATABASE')


# Delete functions: remove from the database some information
def delete_problem(session, problem_name):
    """Delete a problem from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    problem_name : str
        The name of the problem to remove.
    """
    problem = select_problem_by_name(session, problem_name)
    if problem is None:
        raise NoResultFound('No result found for "{}" in Problem table'
                            .format(problem_name))
    for event in problem.events:
        delete_event(session, event.name)
    session.delete(problem)
    session.commit()


def delete_event(session, event_name):
    """Delete an event from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str
        The name of the event to delete.
    """
    event = select_event_by_name(session, event_name)
    submissions = select_submissions_by_state(session, event_name, state=None)
    for sub in submissions:
        delete_submission_similarity(session, sub.id)
    session.delete(event)
    session.commit()


def delete_submission_similarity(session, submission_id):
    """Delete the submission similarity associated with a submission.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    submission_id: int
        The id of the submission to use.
    """
    submission = select_submission_by_id(session, submission_id)
    similarities = []
    similarities += select_similarities_by_target(session, submission)
    similarities += select_similarities_by_source(session, submission)
    for similarity in similarities:
        session.delete(similarity)
    session.commit()


# Add functions: add to the database some information
def add_workflow(session, workflow_object):
    """Add a new workflow.

    Workflow class should exist in ``rampwf.workflows``. The name of the
    workflow will be the classname (e.g. Classifier). Element names are taken
    from ``workflow.element_names``. Element types are inferred from the
    extension. This is important because e.g. the max size and the editability
    will depend on the type.

    ``add_workflow`` is called by :func:`add_problem`, taking the workflow to
    add from the ``problem.py`` file of the starting kit.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    workflow_object : :mod:`rampwf.workflows`
        A ramp workflow instance. Refer to :mod:`rampwf.workflows` for all
        available workflows.
    """
    workflow_name = workflow_object.__class__.__name__
    workflow = select_workflow_by_name(session, workflow_name)
    if workflow is None:
        session.add(Workflow(name=workflow_name))
        workflow = select_workflow_by_name(session, workflow_name)
    for element_name in workflow_object.element_names:
        tokens = element_name.split('.')
        element_filename = tokens[0]
        # inferring that file is code if there is no extension
        element_file_extension_name = ('.'.join(tokens[1:])
                                       if len(tokens) > 1 else 'py')
        extension = select_extension_by_name(session,
                                             element_file_extension_name)
        if extension is None:
            raise ValueError('Unknown extension {}.'
                             .format(element_file_extension_name))
        type_extension = select_submission_type_extension_by_extension(
            session, extension
        )
        if type_extension is None:
            raise ValueError('Unknown file type {}.'
                             .format(element_file_extension_name))

        workflow_element_type = select_workflow_element_type_by_name(
            session, element_filename
        )
        if workflow_element_type is None:
            workflow_element_type = WorkflowElementType(
                name=element_filename, type=type_extension.type
            )
            logger.info('Adding {}'.format(workflow_element_type))
            session.add(workflow_element_type)
        workflow_element = select_workflow_element_by_workflow_and_type(
            session, workflow=workflow,
            workflow_element_type=workflow_element_type
        )
        if workflow_element is None:
            workflow_element = WorkflowElement(
                workflow=workflow,
                workflow_element_type=workflow_element_type
            )
            logger.info('Adding {}'.format(workflow_element))
            session.add(workflow_element)
    session.commit()


def add_problem(session, problem_name, kit_dir, data_dir, force=False):
    """Add a RAMP problem to the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    problem_name : str
        The name of the problem to register in the database.
    kit_dir : str
        The directory where the RAMP kit are located. It will corresponds to
        the key `ramp_kit_dir` of the dictionary created with
        :func:`ramp_utils.generate_ramp_config`.
    data_dir : str
        The directory where the RAMP data are located. It will corresponds to
        the key `ramp_data_dir` of the dictionary created with
        :func:`ramp_utils.generate_ramp_config`.
    force : bool, default is False
        Whether to force add the problem. If ``force=False``, an error is
        raised if the problem was already in the database.
    """
    problem = select_problem_by_name(session, problem_name)
    problem_kit_path = kit_dir
    if problem is not None:
        if not force:
            raise ValueError('Attempting to overwrite a problem and '
                             'delete all linked events. Use"force=True" '
                             'if you want to overwrite the problem and '
                             'delete the events.')
        delete_problem(session, problem_name)

    # load the module to get the type of workflow used for the problem
    problem_module = import_module_from_source(
        os.path.join(problem_kit_path, 'problem.py'), 'problem')
    add_workflow(session, problem_module.workflow)
    problem = Problem(name=problem_name, path_ramp_kit=kit_dir,
                      path_ramp_data=data_dir, session=session)
    logger.info('Adding {}'.format(problem))
    session.add(problem)
    session.commit()


def add_event(session, problem_name, event_name, event_title,
              ramp_sandbox_name, ramp_submissions_path, is_public=False,
              force=False):
    """Add a RAMP event in the database.

    Event file should be set up in ``databoard/specific/events/<event_name>``.
    Should be preceded by adding a problem (cf., :func:`add_problem`), then
    ``problem_name`` imported in the event file (``problem_name`` is acting as
    a pointer for the join). Also adds CV folds.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    problem_name : str
        The problem name associated with the event.
    event_name : str
        The event name.
    event_title : str
        The even title.
    ramp_sandbox_name : str
        Name of the submission which will be considered the sandbox. It will
        correspond to the key ``sandbox_name`` of the dictionary created with
        :func:`ramp_utils.generate_ramp_config`.
    ramp_submissions_path : str
        Path to the deployment RAMP submissions directory. It will corresponds
        to the key `ramp_submissions_dir` of the dictionary created with
        :func:`ramp_utils.generate_ramp_config`.
    is_public : bool, default is False
        Whether the event is made public or not.
    force : bool, default is False
        Whether to overwrite an existing event. If ``false=False``, an error
        will be raised.

    Returns
    -------
    event : Event
        The event which has been registered in the database.
    """
    event = select_event_by_name(session, event_name)
    if event is not None:
        if not force:
            raise ValueError("Attempting to overwrite existing event. "
                             "Use force=True to overwrite.")
        delete_event(session, event_name)

    if event_name[:len(problem_name)+1] != (problem_name + '_'):
        raise ValueError(
            "The event name should start with the problem name: '{}_'. Please "
            "edit the entry <event_name> of the event configuration file "
            .format(problem_name)
        )

    event = Event(name=event_name, problem_name=problem_name,
                  event_title=event_title,
                  ramp_sandbox_name=ramp_sandbox_name,
                  path_ramp_submissions=ramp_submissions_path,
                  session=session)
    event.is_public = is_public
    event.is_send_submitted_mails = False
    event.is_send_trained_mails = False
    logger.info('Adding {}'.format(event))
    session.add(event)
    session.commit()

    X_train, y_train = event.problem.get_train_data()
    cv = event.problem.module.get_cv(X_train, y_train)
    for train_indices, test_indices in cv:
        cv_fold = CVFold(event=event,
                         train_is=train_indices,
                         test_is=test_indices)
        session.add(cv_fold)

    score_types = event.problem.module.score_types
    for score_type in score_types:
        event_score_type = EventScoreType(event=event,
                                          score_type_object=score_type)
        session.add(event_score_type)
    event.official_score_name = score_types[0].name
    session.commit()
    return event


def add_event_admin(session, event_name, user_name):
    """Add an administrator event.

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
    event_admin = select_event_admin_by_instance(session, event, user)
    if event_admin is None:
        event_admin = EventAdmin(event=event, admin=user)
        session.commit()


def add_keyword(session, name, keyword_type, category=None, description=None,
                force=False):
    """Add a keyword to the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str
        The name of the keyword.
    keyword_type : {'data_domain', 'data_science_theme'}
        The type of keyword.
    category : None or str, default is None
        The category of the keyword.
    description : None or str, default is None
        The description of the keyword.
    force : bool, default is False
        Whether or not to overwrite the keyword if it already exists.
    """
    keyword = session.query(Keyword).filter_by(name=name).one_or_none()
    if keyword is not None:
        if not force:
            raise ValueError(
                'Attempting to update an existing keyword. Use "force=True"'
                'to overwrite the keyword.'
            )
        keyword.type = keyword_type
        keyword.category = category
        keyword.description = description
    else:
        keyword = Keyword(name=name, type=keyword_type, category=category,
                          description=description)
        session.add(keyword)
    session.commit()


def add_problem_keyword(session, problem_name, keyword_name, description=None,
                        force=False):
    """Add relationship between a keyword and a problem.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    problem_name : str
        The name of the problem.
    keyword_name : str
        The name of the keyword.
    description : None or str, default is None
        A particular description of the keyword of the particular problem.
    force : bool, default is False
        Whether or not to overwrite the relationship.
    """
    problem = select_problem_by_name(session, problem_name)
    keyword = get_keyword_by_name(session, keyword_name)
    problem_keyword = (session.query(ProblemKeyword)
                              .filter_by(problem=problem, keyword=keyword)
                              .one_or_none())
    if problem_keyword is not None:
        if not force:
            raise ValueError(
                'Attempting to update an existing problem-keyword '
                'relationship. Use "force=True" if you want to overwrite the '
                'relationship.'
            )
        problem_keyword.description = description
    else:
        problem_keyword = ProblemKeyword(
            problem=problem, keyword=keyword, description=description
        )
        session.add(problem_keyword)
    session.commit()


# Getter functions: get information from the database
def get_problem(session, problem_name):
    """Get problem from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    problem_name : str or None
        The name of the problem to query. If None, all the problems will be
        queried.

    Returns
    -------
    problem : :class:`ramp_database.model.Problem` or list of \
:class:`ramp_database.model.Problem`
        The queried problem.
    """
    return select_problem_by_name(session, problem_name)


def get_workflow(session, workflow_name):
    """Get workflow from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    workflow_name : str or None
        The name of the workflow to query. If None, all the workflows will be
        queried.

    Returns
    -------
    workflow : :class:`ramp_database.model.Workflow` or list of \
:class:`ramp_database.model.Workflow`
        The queried workflow.
    """
    return select_workflow_by_name(session, workflow_name)


def get_event(session, event_name):
    """Get event from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str or None
        The name of the event to query. If None, all the events will be
        queried.

    Returns
    -------
    even : :class:`ramp_database.model.Event` or \
list of :class:`ramp_database.model.Event`
        The queried problem.
    """
    return select_event_by_name(session, event_name)


def get_cv_fold_by_event(session, event):
    """Get ScoreType from the database

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str or None
        The event class to query.

    Returns
    -------
    cv fold : : list of all cv folds of this event
    """
    return (session.query(CVFold)
                   .filter(EventScoreType.event_id ==
                           event.id)
                   .all())


def get_score_type_by_event(session, event):
    """Get ScoreType from the database

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    event_name : str or None
        The event class to query

    Returns
    -------
    score type : :class:`ramp_database.model.ScoreType` or \
    list of list of :class:`ramp_database.model.ScoreType`
        The queried problem.
    """
    return (session.query(EventScoreType)
                   .filter(EventScoreType.event_id ==
                           event.id)
                   .all())


def get_event_admin(session, event_name, user_name):
    """Get an administrator event.

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
    event_admin : :class:`ramp_database.model.EventAdmin` or None
        The event/admin instance queried.
    """
    event = select_event_by_name(session, event_name)
    user = select_user_by_name(session, user_name)
    return select_event_admin_by_instance(session, event, user)


def get_keyword_by_name(session, name):
    """Get the keyword filtering by there name

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    name : str or None
        The name of the keyword. If None, all keywords will be returned.

    Returns
    -------
    keyword : :class:`ramp_database.model.Keyword` or  list of \
:class:`ramp.model.Keyword`
        The keyword which have been queried.
    """
    q = session.query(Keyword)
    if name is None:
        return q.all()
    return q.filter_by(name=name).one_or_none()


def get_problem_keyword_by_name(session, problem_name, keyword_name):
    """Get a problem-keyword relationship given their names.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    problem_name : str
        The name of the problem.
    keyword_name : str
        The name of the keyword.

    Returns
    -------
    problem_keyword : :class:`ramp_database.model.ProblemKeyword`
        The problem-keyword relationship.
    """
    problem = select_problem_by_name(session, problem_name)
    keyword = session.query(Keyword).filter_by(name=keyword_name).one_or_none()
    return (session.query(ProblemKeyword)
                   .filter_by(problem=problem, keyword=keyword)
                   .one_or_none())
