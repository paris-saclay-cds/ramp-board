import os

from sqlalchemy import Float
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship

from ramputils.utils import import_module_from_source
from ramputils.utils import encode_string

from .base import Model
from .base import get_deployment_path
from .workflow import Workflow

# TODO: This will be really wrong at some point.
# TODO: We need to pass the configuration or the path to the data instead.
DEPLOYMENT_PATH = get_deployment_path()
RAMP_KITS_PATH = os.path.join(
    DEPLOYMENT_PATH, os.getenv('RAMP_KITS_DIR', 'ramp-kits'))
RAMP_DATA_PATH = os.path.join(
    DEPLOYMENT_PATH, os.getenv('RAMP_DATA_DIR', 'ramp-data'))


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
    events : list of :class:`rampdb.model.Event`
        A back-reference to the event.
    keywords : list of :class:`rampdb.model.ProblemKeyword`
        A back-reference to the keywords associated with the problem.
    """
    __tablename__ = 'problems'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)

    workflow_id = Column(
        Integer, ForeignKey('workflows.id'), nullable=False)
    workflow = relationship(
        'Workflow', backref=backref('problems'))

    def __init__(self, name, session=None):
        self.name = name
        self.reset(session)
        # to check if the module and all required fields are there
        self.module
        self.Predictions
        self.workflow_object

    def __repr__(self):
        return 'Problem({})\n{}'.format(encode_string(self.name),
                                        self.workflow)

<<<<<<< HEAD
<<<<<<< HEAD
    def reset(self, session):
        if session is not None:
            self.workflow = \
                (session.query(Workflow)
                        .filter(Workflow.name == type(self.module.workflow)
                        .__name__).one())
        else:
            self.workflow = \
                (Workflow.query
                         .filter_by(name=type(self.module.workflow).__name__)
                         .one())
=======
=======
>>>>>>> origin/master
    def reset(self):
        self.workflow = Workflow.query.filter_by(
            name=type(self.module.workflow).__name__).one()
>>>>>>> DOC: document the database model (#116)

    @property
    def module(self):
        """module: Get the problem module."""
        return import_module_from_source(
            os.path.join(RAMP_KITS_PATH, self.name, 'problem.py'),
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
        path = os.path.join(RAMP_DATA_PATH, self.name)
        return self.module.get_train_data(path=path)

    def get_test_data(self):
        """The testing data."""
        path = os.path.join(RAMP_DATA_PATH, self.name)
        return self.module.get_test_data(path=path)

    def ground_truths_train(self):
        """Predictions: The true labels for the training."""
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
    submission_id = Column(
        Integer, ForeignKey('submissions.id'))
    submission = relationship('Submission', backref=backref(
        'historical_contributivitys', cascade='all, delete-orphan'))

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
    problems : list of :class:`rampdb.model.ProblemKeyword`
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
    problem : :class:`rampdb.model.Problem`
        The problem instance.
    keyword_id : int
        The ID of the keyword.
    keyword : :class:`rampdb.model.Keyword`
        The keyword instance.
    """
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
