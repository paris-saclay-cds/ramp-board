import logging
import os

from sqlalchemy.orm.exc import NoResultFound

from ramputils.utils import import_module_from_source

from ._query import select_event_by_name
from ._query import select_extension_by_name
from ._query import select_problem_by_name
from ._query import select_submissions_by_state
from ._query import select_similarities_by_source
from ._query import select_similarities_by_target
from ._query import select_submission_by_id
from ._query import select_submission_type_extension_by_extension
from ._query import select_workflow_by_name
from ._query import select_workflow_element_by_workflow_and_type
from ._query import select_workflow_element_type_by_name

from ..model import Problem
from ..model import Workflow
from ..model import WorkflowElement
from ..model import WorkflowElementType

logger = logging.getLogger('DATABASE')


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
    for sub_id, _, _ in submissions:
        delete_submission_similarity(session, sub_id)
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
        if len(tokens) > 2:
            raise ValueError('File name {} should contain at most one "."'
                             .format(element_name))
        element_file_extension_name = tokens[1] if len(tokens) == 2 else 'py'
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


def add_problem(session, problem_name, kits_dir, force=False):
    """Add a RAMP problem to the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    problem_name : str
        The name of the problem to register in the database.
    kits_dir : str
        The directory where the RAMP kits are located.
    force : bool, default is False
        Whether to force add the problem. If ``force=False``, an error is
        raised if the problem was already in the database.
    """
    problem = select_problem_by_name(session, problem_name)
    problem_kits_path = os.path.join(kits_dir, problem_name)
    if problem is not None and not force:
        if not force:
            raise ValueError('Attempting to overwrite a problem and '
                             'delete all linked events. Use"force=True" '
                             'if you want to overwrite the problem and '
                             'delete the events.')
        delete_problem(session, problem_name)

    # load the module to get the type of workflow used for the problem
    problem_module = import_module_from_source(
        os.path.join(problem_kits_path, 'problem.py'), 'problem')
    add_workflow(session, problem_module.workflow)
    problem = Problem(name=problem_name, session=session)
    logger.info('Adding {}'.format(problem))
    session.add(problem)
    session.commit()


# Getter functions: get information from the database
def get_problem(session, problem_name):
    """Get problem from the database.

    Parameters
    ----------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    problem_name : str or None
        The name of the problem to remove. If None, all the problems will be
        queried.

    Returns
    -------
    problem : :class:`rampdb.model.Problem` or list of :class:`rampdb.model.Problem`
        The queried problem.
    """
    return select_problem_by_name(session, problem_name)