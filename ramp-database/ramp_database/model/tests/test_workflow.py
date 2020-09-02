import re
import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Model
from ramp_database.model import Problem
from ramp_database.model import SubmissionFile
from ramp_database.model import SubmissionFileType
from ramp_database.model import Workflow
from ramp_database.model import WorkflowElement
from ramp_database.model import WorkflowElementType

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.event import get_workflow


@pytest.fixture(scope='module')
def session_scope_module(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_workflow_element_type_model(session_scope_module):
    workflow_element_type = \
        (session_scope_module.query(WorkflowElementType)
                             .filter(WorkflowElementType.type_id ==
                                     SubmissionFileType.id)
                             .filter(SubmissionFileType.name == 'code')
                             .first())

    assert workflow_element_type.file_type == 'code'
    assert workflow_element_type.is_editable is True
    assert workflow_element_type.max_size == 100000
    assert re.match(r"WorkflowElementType\(.*\)",
                    repr(workflow_element_type))


@pytest.mark.parametrize(
    'backref, expected_type',
    [('workflows', WorkflowElement)]
)
def test_workflow_element_type_model_backref(session_scope_module, backref,
                                             expected_type):
    workflow_element_type = \
        (session_scope_module.query(WorkflowElementType)
                             .filter(WorkflowElementType.type_id ==
                                     SubmissionFileType.id)
                             .filter(SubmissionFileType.name == 'code')
                             .first())
    backref_attr = getattr(workflow_element_type, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)


def test_workflow_model(session_scope_module):
    workflow = get_workflow(session_scope_module, 'Estimator')
    assert re.match(r'Workflow\(.*\)\n\t.*WorkflowElement.*', repr(workflow))


@pytest.mark.parametrize(
    'backref, expected_type',
    [('problems', Problem),
     ('elements', WorkflowElement)]
)
def test_workflow_model_backref(session_scope_module, backref, expected_type):
    workflow = get_workflow(session_scope_module, 'Estimator')
    backref_attr = getattr(workflow, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)


def test_workflow_element_model(session_scope_module):
    workflow_element = \
        (session_scope_module.query(WorkflowElement)
                             .filter(WorkflowElement.workflow_id ==
                                     Workflow.id)
                             .filter(Workflow.name == 'Estimator')
                             .one())

    assert re.match(r'Workflow\(.*\): WorkflowElement\(.*\)',
                    repr(workflow_element))
    assert workflow_element.type == 'estimator'
    assert workflow_element.file_type == 'code'
    assert workflow_element.is_editable is True
    assert workflow_element.max_size == 100000


@pytest.mark.parametrize(
    'backref, expected_type',
    [('submission_files', SubmissionFile)]
)
def test_workflow_element_model_backref(session_scope_module, backref,
                                        expected_type):
    workflow_element = \
        (session_scope_module.query(WorkflowElement)
                             .filter(WorkflowElement.workflow_id ==
                                     Workflow.id)
                             .filter(Workflow.name == 'Estimator')
                             .one())
    backref_attr = getattr(workflow_element, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)
