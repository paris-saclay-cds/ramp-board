from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship

from .base import Model

__all__ = [
    'Workflow',
    'WorkflowElement',
    'WorkflowElementType',
]


class WorkflowElementType(Model):
    __tablename__ = 'workflow_element_types'

    id = Column(Integer, primary_key=True)
    # file name without extension
    # eg, regressor, classifier, external_data
    name = Column(String, nullable=False, unique=True)

    # eg, code, text, data
    type_id = Column(
        Integer, ForeignKey('submission_file_types.id'), nullable=False)
    type = relationship(
        'SubmissionFileType', backref=backref('workflow_element_types'))

    def __repr__(self):
        text = 'WorkflowElementType(name={}, type={}'.format(
            self.name, self.type.name)
        text += 'is_editable={}, max_size={})'.format(
            self.type.is_editable, self.type.max_size)
        return text

    @property
    def file_type(self):
        return self.type.name

    @property
    def is_editable(self):
        return self.type.is_editable

    @property
    def max_size(self):
        return self.type.max_size


# training and test code now belongs to the workflow, not the workflow
# element. This latter would requre to carefully define workflow element
# interfaces. Eg, a dilemma: classifier + calibrator needs to handled at the
# workflow level (since calibrator needs held out data). Eventually we should
# have both workflow-level and workflow-element-level code to avoid code
# repetiotion.
class Workflow(Model):
    __tablename__ = 'workflows'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)

    def __init__(self, name):
        self.name = name
        # to check if the module and all required fields are there

    def __repr__(self):
        text = 'Workflow({})'.format(self.name)
        for workflow_element in self.elements:
            text += '\n\t' + str(workflow_element)
        return text


# In lists we will order files according to their ids
# many-to-many link
# For now files define the workflow, so eg, a feature_extractor + regressor
# is not the same workflow as a feature_extractor + regressor + external data,
# even though the training codes are the same.
class WorkflowElement(Model):
    __tablename__ = 'workflow_elements'

    id = Column(Integer, primary_key=True)
    # Normally name will be the same as workflow_element_type.type.name,
    # unless specified otherwise. It's because in more complex workflows
    # the same type can occur more then once. self.type below will always
    # refer to workflow_element_type.type.name
    name = Column(String, nullable=False)

    workflow_id = Column(
        Integer, ForeignKey('workflows.id'))
    workflow = relationship(
        'Workflow', backref=backref('elements'))

    workflow_element_type_id = Column(
        Integer, ForeignKey('workflow_element_types.id'),
        nullable=False)
    workflow_element_type = relationship(
        'WorkflowElementType', backref=backref('workflows'))

    def __init__(self, workflow, workflow_element_type, name_in_workflow=None):
        self.workflow = workflow
        self.workflow_element_type = workflow_element_type
        if name_in_workflow is None:
            self.name = self.workflow_element_type.name
        else:
            self.name = name_in_workflow

    def __repr__(self):
        return 'Workflow({}): WorkflowElement({})'.format(
            self.workflow.name, self.name)

    # e.g. 'regression', 'external_data'. Normally == name
    @property
    def type(self):
        return self.workflow_element_type.name

    @property
    def file_type(self):
        return self.workflow_element_type.file_type

    @property
    def is_editable(self):
        return self.workflow_element_type.is_editable

    @property
    def max_size(self):
        return self.workflow_element_type.max_size
