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
    """WorkflowElementType table.

    Attributes
    ----------
    id : int
        The ID of the table row.
    name : str
        The name of the workflow element type.
    type_id : int
        The ID of the submission file type.
    type : :class:`ramp_database.model.SubmissionFileType`
        The submission file type instance.
    workflows : list of :class:`ramp_database.model.WorkflowElement`
        A back-reference to the workflows linked with the workflow element.
    """
    __tablename__ = 'workflow_element_types'

    id = Column(Integer, primary_key=True)
    # file name without extension
    # eg, regressor, classifier, external_data
    name = Column(String, nullable=False, unique=True)

    # eg, code, text, data
    type_id = Column(Integer, ForeignKey('submission_file_types.id'),
                     nullable=False)
    type = relationship('SubmissionFileType',
                        backref=backref('workflow_element_types'))

    def __repr__(self):
        return (
            'WorkflowElementType(name={}, type={} is_editable={}, max_size={})'
            .format(self.name, self.type.name, self.type.is_editable,
                    self.type.max_size)
        )

    @property
    def file_type(self):
        """str: Name of the submission file type."""
        return self.type.name

    @property
    def is_editable(self):
        """bool: Whether or not the submission file is an editable type on the
        frontend."""
        return self.type.is_editable

    @property
    def max_size(self):
        """int: The maximum size of supported by the submission file type."""
        return self.type.max_size


# training and test code now belongs to the workflow, not the workflow
# element. This latter would requre to carefully define workflow element
# interfaces. Eg, a dilemma: classifier + calibrator needs to handled at the
# workflow level (since calibrator needs held out data). Eventually we should
# have both workflow-level and workflow-element-level code to avoid code
# repetiotion.
class Workflow(Model):
    """Workflow table.

    Parameters
    ----------
    name : str
        The name of the workflow.

    Attributes
    ----------
    id : int
        The ID of the table row.
    name : str
        The name of the workflow.
    problems : list of :class:`ramp_database.model.Problem`
        A back-reference to the problems using this workflow.
    elements : list of :class:`ramp_database.model.WorkflowElement`
        A back-reference to the elements of the workflow.
    """
    __tablename__ = 'workflows'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)

    def __init__(self, name):
        self.name = name

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
    """WorkflowElement table.

    Parameters
    ----------
    workflow : :class:`ramp_database.model.Workflow`
        A workflow instance.
    workflow_element_type : :class:`ramp_database.model.WorkflowElementType`
        A workflow element type instance.
    name_in_workflow : None or str, default is None

    Attributes
    ----------
    id : int
        The ID of the table row.
    name : str
        The name of the workflow element.
    workflow_id : int
        The ID of the associated workflow.
    workflow : :class:`ramp_database.model.Workflow`
        The workflow instance.
    workflow_element_type_id : int
        The ID of the associated workflow element type.
    workflow_element_type : :class:`ramp_database.model.WorkflowElementType`
        The workflow element type instance.
    submission_files : list of :class:`ramp_database.model.SubmissionFile`
        A back-reference to the submission file associated with the workflow
        element.
    """
    __tablename__ = 'workflow_elements'

    id = Column(Integer, primary_key=True)
    # Normally name will be the same as workflow_element_type.type.name,
    # unless specified otherwise. It's because in more complex workflows
    # the same type can occur more then once. self.type below will always
    # refer to workflow_element_type.type.name
    name = Column(String, nullable=False)

    workflow_id = Column(Integer, ForeignKey('workflows.id'))
    workflow = relationship('Workflow', backref=backref('elements'))

    workflow_element_type_id = Column(Integer,
                                      ForeignKey('workflow_element_types.id'),
                                      nullable=False)
    workflow_element_type = relationship('WorkflowElementType',
                                         backref=backref('workflows'))

    def __init__(self, workflow, workflow_element_type, name_in_workflow=None):
        self.workflow = workflow
        self.workflow_element_type = workflow_element_type
        self.name = (self.workflow_element_type.name
                     if name_in_workflow is None else name_in_workflow)

    def __repr__(self):
        return 'Workflow({}): WorkflowElement({})'.format(self.workflow.name,
                                                          self.name)

    @property
    def type(self):
        """str: Name of the workflow element type."""
        return self.workflow_element_type.name

    @property
    def file_type(self):
        """str: Name of the submission file type."""
        return self.workflow_element_type.file_type

    @property
    def is_editable(self):
        """bool: Whether or not the submission file is an editable type on the
        frontend."""
        return self.workflow_element_type.is_editable

    @property
    def max_size(self):
        """int: The maximum size of supported by the submission file type."""
        return self.workflow_element_type.max_size
