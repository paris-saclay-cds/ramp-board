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

from .base import Model
from .base import encode_string
from .base import get_deployment_path
from .workflow import Workflow

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
    __tablename__ = 'problems'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)

    workflow_id = Column(
        Integer, ForeignKey('workflows.id'), nullable=False)
    workflow = relationship(
        'Workflow', backref=backref('problems'))

    def __init__(self, session, name):
        self.name = name
        self.reset(session)
        # to check if the module and all required fields are there
        self.module
        self.Predictions
        self.workflow_object

    def __repr__(self):
        return 'Problem({})\n{}'.format(
            encode_string(self.name), self.workflow)

    def reset(self, session):
        print(Workflow)
        self.workflow = \
            (session.query(Workflow)
                    .filter(Workflow.name == type(self.module.workflow)
                    .__name__).one())
        # TODO: now that we close the connection we need to pass the session.
        # self.workflow = Workflow.query.filter_by(
        #     name=type(self.module.workflow).__name__).one()

    @property
    def module(self):
        return import_module_from_source(
            os.path.join(RAMP_KITS_PATH, self.name, 'problem.py'),
            'problem'
        )

    @property
    def title(self):
        return self.module.problem_title

    @property
    def Predictions(self):
        return self.module.Predictions

    def get_train_data(self):
        path = os.path.join(RAMP_DATA_PATH, self.name)
        return self.module.get_train_data(path=path)

    def get_test_data(self):
        path = os.path.join(RAMP_DATA_PATH, self.name)
        return self.module.get_test_data(path=path)

    def ground_truths_train(self):
        _, y_train = self.get_train_data()
        return self.Predictions(y_true=y_train)

    def ground_truths_test(self):
        _, y_test = self.get_test_data()
        return self.Predictions(y_true=y_test)

    def ground_truths_valid(self, test_is):
        _, y_train = self.get_train_data()
        return self.Predictions(y_true=y_train[test_is])

    @property
    def workflow_object(self):
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
