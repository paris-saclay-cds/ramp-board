import re
import shutil

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Model
from rampdb.model import Problem
from rampdb.model import SubmissionFileType
from rampdb.model import Workflow
from rampdb.model import WorkflowElement
from rampdb.model import WorkflowElementType

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.event import get_workflow


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture(scope='module')
def session_scope_module(config):
    try:
        create_toy_db(config)
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
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


def test_workflow_model(session_scope_module):
    workflow = get_workflow(session_scope_module, 'Classifier')
    assert re.match(r'Workflow\(.*\)\n\t.*WorkflowElement.*', repr(workflow))


@pytest.mark.parametrize(
    'backref, expected_type',
    [('problems', Problem)]
)
def test_workflow_model_backref(session_scope_module, backref, expected_type):
    workflow = get_workflow(session_scope_module, 'Classifier')
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
                             .filter(Workflow.name == 'Classifier')
                             .one())

    assert re.match(r'Workflow\(.*\): WorkflowElement\(.*\)',
                    repr(workflow_element))
    assert workflow_element.type == 'classifier'
    assert workflow_element.file_type == 'code'
    assert workflow_element.is_editable is True
    assert workflow_element.max_size == 100000
