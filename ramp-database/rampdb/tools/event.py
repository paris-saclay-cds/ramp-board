import logging
import os

from ramputils import generate_ramp_config
from ramputils.utils import import_module_from_source

from ..utils import setup_db

from ._query import select_problem_by_name
from ..model import Problem

logger = logging.getLogger('DATABASE')


# def add_workflow(workflow_object):
#     """Add a new workflow.

#     Workflow class should exist in ``rampwf.workflows``. The name of the
#     workflow will be the classname (e.g. Classifier). Element names are taken
#     from ``workflow.element_names``. Element types are inferred from the
#     extension. This is important because e.g. the max size and the editability
#     will depend on the type.

#     ``add_workflow`` is called by :func:`add_problem`, taking the workflow to
#     add from the ``problem.py`` file of the starting kit.

#     Parameters
#     ----------
#     workflow_object : ramp.workflows
#         A ramp workflow instance.
#     """
#     workflow_name = workflow_object.__class__.__name__
#     workflow = Workflow.query.filter_by(name=workflow_name).one_or_none()
#     if workflow is None:
#         db.session.add(Workflow(name=workflow_name))
#         workflow = Workflow.query.filter_by(name=workflow_name).one()
#     for element_name in workflow_object.element_names:
#         tokens = element_name.split('.')
#         element_filename = tokens[0]
#         # inferring that file is code if there is no extension
#         if len(tokens) > 2:
#             raise ValueError('File name {} should contain at most one "."'
#                              .format(element_name))
#         element_file_extension_name = tokens[1] if len(tokens) == 2 else 'py'
#         extension = Extension.query.filter_by(
#             name=element_file_extension_name).one_or_none()
#         if extension is None:
#             raise ValueError('Unknown extension {}.'
#                              .format(element_file_extension_name))
#         type_extension = SubmissionFileTypeExtension.query.filter_by(
#             extension=extension).one_or_none()
#         if type_extension is None:
#             raise ValueError('Unknown file type {}.'
#                              .format(element_file_extension_name))

#         workflow_element_type = WorkflowElementType.query.filter_by(
#             name=element_filename).one_or_none()
#         if workflow_element_type is None:
#             workflow_element_type = WorkflowElementType(
#                 name=element_filename, type=type_extension.type)
#             logger.info('Adding {}'.format(workflow_element_type))
#             db.session.add(workflow_element_type)
#             db.session.commit()
#         workflow_element = WorkflowElement.query.filter_by(
#             workflow=workflow,
#             workflow_element_type=workflow_element_type).one_or_none()
#         if workflow_element is None:
#             workflow_element = WorkflowElement(
#                 workflow=workflow,
#                 workflow_element_type=workflow_element_type)
#             logger.info('Adding {}'.format(workflow_element))
#             db.session.add(workflow_element)
#     db.session.commit()


def add_problem(config, problem_name, force=False):
    """Add a RAMP problem to the database.

    Parameters
    ----------
    config : dict
        The overall configuration file.
    problem_name : str
        The name of the problem to register in the database.
    force : bool, default is False
        Whether to force add the problem. If ``force=False``, an error is
        raised if the problem was already in the database.
    """
    database_config = config['sqlalchemy']
    ramp_config = generate_ramp_config(config)
    db, Session = setup_db(database_config)
    with db.connect() as conn:
        session = Session(bind=conn)

        problem = select_problem_by_name(session, problem_name)
        problem_kits_path = os.path.join(ramp_config['ramp_kits_dir'],
                                         problem_name)
        if problem is not None and not force:
            if not force:
                raise ValueError('Attempting to overwrite a problem and '
                                'delete all linked events. Use"force=True" '
                                'if you want to overwrite the problem and '
                                'delete the events.')
            delete_problem(problem_name)

        # load the module to get the type of workflow used for the problem
        problem_module = import_module_from_source(
            os.path.join(problem_kits_path, 'problem.py'), 'problem')
        add_workflow(problem_module.workflow)
        problem = Problem(name=problem_name)
        logger.info('Adding {}'.format(problem))
        db.session.add(problem)
        db.session.commit()