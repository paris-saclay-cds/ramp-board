import os

from sqlalchemy import Float
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship

from ramp_utils.utils import import_module_from_source

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
    path_ramp_kit : str
        The path where the kit is located. It will corresponds to
        the key `ramp_kit_dir` of the dictionary created with
        :func:`ramp_utils.generate_ramp_config`.
    path_ramp_data : str
        The path where the data are located. It will corresponds to
        the key `ramp_data_dir` of the dictionary created with
        :func:`ramp_utils.generate_ramp_config`.
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
    workflow : :class:`ramp_database.model.Worflow`
        The workflow instance.
    path_ramp_kit : str
        The path where the kit are located.
    path_ramp_data : str
        The path where the data are located.
    events : list of :class:`ramp_database.model.Event`
        A back-reference to the event.
    keywords : list of :class:`ramp_database.model.ProblemKeyword`
        A back-reference to the keywords associated with the problem.
    """
    __tablename__ = 'problems'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)

    workflow_id = Column(Integer, ForeignKey('workflows.id'), nullable=False)
    workflow = relationship('Workflow', backref=backref('problems'))

    # XXX: big change in the database
    path_ramp_kit = Column(String, nullable=False, unique=False)
    path_ramp_data = Column(String, nullable=False, unique=False)

    def __init__(self, name, path_ramp_kit, path_ramp_data, session=None):
        self.name = name
        self.path_ramp_kit = path_ramp_kit
        self.path_ramp_data = path_ramp_data
        self.reset(session)

    def __repr__(self):
        return 'Problem({})\n{}'.format(self.name, self.workflow)

    def reset(self, session):
        """Reset the workflow."""
        if session is not None:
            self.workflow = \
                (session.query(Workflow)
                        .filter(Workflow.name ==                      # noqa
                                type(self.module.workflow).__name__)  # noqa
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
            os.path.join(self.path_ramp_kit, 'problem.py'),
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
        """The training data."""
        return self.module.get_train_data(path=self.path_ramp_data)

    def get_test_data(self):
        """The testing data."""
        return self.module.get_test_data(path=self.path_ramp_data)

    def ground_truths_train(self):
        """Predictions: the true labels for the training."""
        _, y_train = self.get_train_data()
        return self.Predictions(y_true=y_train)

    def ground_truths_test(self):
        """Predictions: the true labels for the testing."""
        _, y_test = self.get_test_data()
        return self.Predictions(y_true=y_test)

    def ground_truths_valid(self, test_is):
        """Predictions: the true labels for the validation."""
        _, y_train = self.get_train_data()
        return self.Predictions(y_true=y_train[test_is])

    @property
    def workflow_object(self):
        """:class:`rampwf.worflows`: The workflow instance used for the
        problem."""
        return self.module.workflow


class HistoricalContributivity(Model):
    """HistoricalContributivity table.

    Attributes
    ----------
    id : int
        The ID of the table row.
    timestamp : datetime
        The date and time of the submission.
    submission_id : int
        The ID of the submission.
    submission : :class:`rampwf.model.Submission`
        The submission instance.
    contributivity : float
        The contributivity of the current submission.
    historical_contributivity : float
        The historical contributivity.
    """
    __tablename__ = 'historical_contributivity'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    submission_id = Column(Integer, ForeignKey('submissions.id'))
    submission = relationship('Submission',
                              backref=backref('historical_contributivitys',
                                              cascade='all, delete-orphan'))

    contributivity = Column(Float, default=0.0)
    historical_contributivity = Column(Float, default=0.0)


class Keyword(Model):
    """Keyword table.

    Attributes
    ----------
    id : int
        The ID of the table row.
    name : str
        The keyword name.
    type : {'data_domain', 'data_science_theme'}
        The type of keyword.
    category : str
        The category of the keyword.
    description : str
        The description of the keyword.
    problems : list of :class:`ramp_database.model.ProblemKeyword`
        A back-reference to the problems associated with the keyword.
    """
    __tablename__ = 'keywords'

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(), nullable=False, unique=True)
    # 'data_domain' or 'data_science_theme'
    type = Column(String(), nullable=False)
    # 'industrial', 'research', etc.
    category = Column(String())
    description = Column(String())


class ProblemKeyword(Model):
    """ProblemKeyword table.

    This a many-to-many relationship between a Problem and a Keyword.

    Attributes
    ----------
    id : int
        The ID of the table row.
    description : str
        Optional description of the keyword for a particular problem.
    problem_id : int
        The ID of the problem.
    problem : :class:`ramp_database.model.Problem`
        The problem instance.
    keyword_id : int
        The ID of the keyword.
    keyword : :class:`ramp_database.model.Keyword`
        The keyword instance.
    """
    __tablename__ = 'problem_keywords'

    id = Column(Integer, primary_key=True)
    description = Column(String)

    problem_id = Column(Integer, ForeignKey('problems.id'), nullable=False)
    problem = relationship('Problem',
                           backref=backref('keywords',
                                           cascade='all, delete-orphan'))

    keyword_id = Column(Integer, ForeignKey('keywords.id'), nullable=False)
    keyword = relationship('Keyword',
                           backref=backref('problems',
                                           cascade='all, delete-orphan'))
