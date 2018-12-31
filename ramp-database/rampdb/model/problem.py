import os

from sqlalchemy import Float
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship

from ramputils import import_module_from_source
from ramputils import encode_string

from .base import Model
from .workflow import Workflow


__all__ = [
    'Problem',
    'HistoricalContributivity',
    'Keyword',
    'ProblemKeyword',
]


class Problem(Model):
    """Problem table.

    Parameters
    ----------
    name : str
        The name of the problem.
    path_ramp_kits : str
        The path where the kits are located. It will corresponds to
        the key `ramp_kits_dir` of the dictionary created with
        :func:`ramputils.generate_ramp_config`.
    path_ramp_data : str
        The path where the data are located. It will corresponds to
        the key `ramp_data_dir` of the dictionary created with
        :func:`ramputils.generate_ramp_config`.
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.

    Attributes
    ----------
    id : int
        The ID of the table row.
    name : str
        The name of the problem.
    workflow_id : id
        The ID of the associated workflow.
    workflow : :class:`rampdb.model.Worflow`
        The workflow instance.
    path_ramp_kits : str
        The path where the kits are located.
    path_ramp_data : str
        The path where the data are located.
    """
    __tablename__ = 'problems'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)

    workflow_id = Column(Integer, ForeignKey('workflows.id'), nullable=False)
    workflow = relationship('Workflow', backref=backref('problems'))

    # XXX: big change in the database
    path_ramp_kits = Column(String, nullable=False, unique=False)
    path_ramp_data = Column(String, nullable=False, unique=False)

    def __init__(self, name, path_ramp_kits, path_ramp_data, session=None):
        self.name = name
        self.path_ramp_kits = path_ramp_kits
        self.path_ramp_data = path_ramp_data
        self.reset(session)

    def __repr__(self):
        return 'Problem({})\n{}'.format(encode_string(self.name),
                                        self.workflow)

    def reset(self, session):
        """Reset the workflow."""
        if session is not None:
            self.workflow = \
                (session.query(Workflow)
                        .filter(Workflow.name ==
                                type(self.module.workflow).__name__)
                        .one())
        else:
            self.workflow = \
                (Workflow.query
                         .filter_by(name=type(self.module.workflow).__name__)
                         .one())

    @property
    def module(self):
        """module: Get the problem module."""
        return import_module_from_source(
            os.path.join(self.path_ramp_kits, self.name, 'problem.py'),
            'problem'
        )

    @property
    def title(self):
        """str: The title of the problem."""
        return self.module.problem_title

    @property
    def Predictions(self):
        """:class:`rampwf.prediction_types`: The predictions used for the
        problem."""
        return self.module.Predictions

    def get_train_data(self):
        """ndarray: The training data."""
        path = os.path.join(self.path_ramp_data, self.name)
        return self.module.get_train_data(path=path)

    def get_test_data(self):
        """ndarray: The testing data."""
        path = os.path.join(self.path_ramp_data, self.name)
        return self.module.get_test_data(path=path)

    def ground_truths_train(self):
        """ndarray: The true labels for the training."""
        _, y_train = self.get_train_data()
        return self.Predictions(y_true=y_train)

    def ground_truths_test(self):
        """ndarray: the true labels for the testing."""
        _, y_test = self.get_test_data()
        return self.Predictions(y_true=y_test)

    def ground_truths_valid(self, test_is):
        """ndarray: the true labels for the validation."""
        _, y_train = self.get_train_data()
        return self.Predictions(y_true=y_train[test_is])

    @property
    def workflow_object(self):
        """:class:`rampwf.worflows`: The workflow instance used for the
        problem."""
        return self.module.workflow


class HistoricalContributivity(Model):
    __tablename__ = 'historical_contributivity'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    submission_id = Column(
        Integer, ForeignKey('submissions.id'))
    submission = relationship('Submission', backref=backref(
        'historical_contributivitys', cascade='all, delete-orphan'))

    contributivity = Column(Float, default=0.0)
    historical_contributivity = Column(Float, default=0.0)


class Keyword(Model):
    __tablename__ = 'keywords'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(), nullable=False, unique=True)
    # 'data_domain' or 'data_science_theme'
    type = Column(String(), nullable=False)
    # 'industrial', 'research', etc.
    category = Column(String())
    description = Column(String())


class ProblemKeyword(Model):
    __tablename__ = 'problem_keywords'

    id = Column(Integer, primary_key=True)
    # optional description of the keyword particular to a problem
    description = Column(String)

    problem_id = Column(
        Integer, ForeignKey('problems.id'), nullable=False)
    problem = relationship(
        'Problem', backref=backref(
            'keywords', cascade='all, delete-orphan'))

    keyword_id = Column(
        Integer, ForeignKey('keywords.id'), nullable=False)
    keyword = relationship(
        'Keyword', backref=backref(
            'problems', cascade='all, delete-orphan'))
