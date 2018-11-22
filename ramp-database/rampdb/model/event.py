import uuid
import datetime

from sqlalchemy import Float
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy import UniqueConstraint
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship

from .base import Model
from .problem import Problem
from .score import ScoreType

__all__ = [
    'Event',
    'EventTeam',
    'EventAdmin',
    'EventScoreType',
]


class Event(Model):
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    title = Column(String, nullable=False)

    problem_id = Column(
        Integer, ForeignKey('problems.id'), nullable=False)
    problem = relationship('Problem', backref=backref(
        'events', cascade='all, delete-orphan'))

    max_members_per_team = Column(Integer, default=1)
    # max number of submissions in Caruana's ensemble
    max_n_ensemble = Column(Integer, default=80)
    is_send_trained_mails = Column(Boolean, default=True)
    is_send_submitted_mails = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)
    is_controled_signup = Column(Boolean, default=True)

    min_duration_between_submissions = Column(Integer, default=15 * 60)
    opening_timestamp = Column(
        DateTime, default=datetime.datetime(2000, 1, 1, 0, 0, 0))
    # before links to submissions in leaderboard are not alive
    public_opening_timestamp = Column(
        DateTime, default=datetime.datetime(2000, 1, 1, 0, 0, 0))
    closing_timestamp = Column(
        DateTime, default=datetime.datetime(2100, 1, 1, 0, 0, 0))

    # the name of the score in self.event_score_types which is used for
    # ensembling and contributivity.
    official_score_name = Column(String)
    # official_score_index = Column(Integer, default=0)

    combined_combined_valid_score = Column(Float, default=None)
    combined_combined_test_score = Column(Float, default=None)
    combined_foldwise_valid_score = Column(Float, default=None)
    combined_foldwise_test_score = Column(Float, default=None)

    n_submissions = Column(Integer, default=0)

    public_leaderboard_html_no_links = Column(String, default=None)
    public_leaderboard_html_with_links = Column(String, default=None)
    private_leaderboard_html = Column(String, default=None)
    failed_leaderboard_html = Column(String, default=None)
    new_leaderboard_html = Column(String, default=None)

    def __init__(self, problem_name, name, event_title, session):
        self.session = session
        self.name = name
        # to check if the module and all required fields are there
        # db fields are later initialized by tools._set_table_attribute
        self.problem = Problem.query.filter_by(name=problem_name).one()
        self.title = event_title
        self.Predictions

    def __repr__(self):
        repr = 'Event({})'.format(self.name)
        return repr

    def set_n_submissions(self):
        self.n_submissions = 0
        for event_team in self.event_teams:
            # substract one for starting kit
            self.n_submissions += len(event_team.submissions) - 1
        self.session.commit()

    @property
    def Predictions(self):
        return self.problem.Predictions

    @property
    def workflow(self):
        return self.problem.workflow

    @property
    def official_score_type(self):
        return EventScoreType.query.filter_by(
            event=self, name=self.official_score_name).one()

    @property
    def official_score_function(self):
        return self.official_score_type.score_function

    @property
    def combined_combined_valid_score_str(self):
        return None if self.combined_foldwise_valid_score is None else str(
            round(self.combined_combined_valid_score,
                  self.official_score_type.precision))

    @property
    def combined_combined_test_score_str(self):
        return None if self.combined_combined_test_score is None else str(
            round(self.combined_combined_test_score,
                  self.official_score_type.precision))

    @property
    def combined_foldwise_valid_score_str(self):
        return None if self.combined_foldwise_valid_score is None else str(
            round(self.combined_foldwise_valid_score,
                  self.official_score_type.precision))

    @property
    def combined_foldwise_test_score_str(self):
        return None if self.combined_foldwise_test_score is None else str(
            round(self.combined_foldwise_test_score,
                  self.official_score_type.precision))

    @property
    def is_open(self):
        now = datetime.datetime.utcnow()
        return now > self.opening_timestamp and now < self.closing_timestamp

    @property
    def is_public_open(self):
        now = datetime.datetime.utcnow()
        return now > self.public_opening_timestamp\
            and now < self.closing_timestamp

    @property
    def is_closed(self):
        now = datetime.datetime.utcnow()
        return now > self.closing_timestamp

    @property
    def n_jobs(self):
        """Number of jobs for local parallelization.

        return: number of live cv folds.
        """
        return sum(1 for cv_fold in self.cv_folds if cv_fold.type == 'live')

    @property
    def n_participants(self):
        return len(self.event_teams)


class EventScoreType(Model):
    __tablename__ = 'event_score_types'

    id = Column(Integer, primary_key=True)
    # Can be renamed, default is the same as score_type.name
    name = Column(String, nullable=False)

    event_id = Column(
        Integer, ForeignKey('events.id'), nullable=False)
    event = relationship('Event', backref=backref(
        'score_types', cascade='all, delete-orphan'))

    score_type_id = Column(
        Integer, ForeignKey('score_types.id'), nullable=False)
    score_type = relationship(
        'ScoreType', backref=backref('events'))

    # display precision in n_digits
    # default is the same as score_type.precision
    precision = Column(Integer)

    UniqueConstraint(event_id, score_type_id, name='es_constraint')
    UniqueConstraint(event_id, name, name='en_constraint')

    def __init__(self, event, score_type_object):
        self.event = event
        # XXX
        self.score_type = ScoreType(uuid.uuid4(), True, 0, 1)
        # XXX after migration we should store the index of the
        # score_type so self.score_type_object (should be renamed
        # score_type) wouldn't have to do a search each time.
        self.name = score_type_object.name
        self.precision = score_type_object.precision
        self.score_type_object
        self.score_function
        self.is_lower_the_better
        self.minimum
        self.maximum
        self.worst

    def __repr__(self):
        repr = '{}: {}'.format(self.name, self.event)
        return repr

    @property
    def score_type_object(self):
        score_types = self.event.problem.module.score_types
        for score_type in score_types:
            if score_type.name == self.name:
                return score_type

    @property
    def score_function(self):
        return self.score_type_object.score_function

    @property
    def is_lower_the_better(self):
        return self.score_type_object.is_lower_the_better

    @property
    def minimum(self):
        return self.score_type_object.minimum

    @property
    def maximum(self):
        return self.score_type_object.maximum

    @property
    def worst(self):
        return self.score_type_object.worst


class EventAdmin(Model):
    __tablename__ = 'event_admins'

    id = Column(Integer, primary_key=True)

    event_id = Column(
        Integer, ForeignKey('events.id'), nullable=False)
    event = relationship('Event', backref=backref(
        'event_admins', cascade='all, delete-orphan'))

    admin_id = Column(
        Integer, ForeignKey('users.id'), nullable=False)
    admin = relationship(
        'User', backref=backref('admined_events'))


# many-to-many
class EventTeam(Model):
    __tablename__ = 'event_teams'

    id = Column(Integer, primary_key=True)

    event_id = Column(
        Integer, ForeignKey('events.id'), nullable=False)
    event = relationship('Event', backref=backref(
        'event_teams', cascade='all, delete-orphan'))

    team_id = Column(
        Integer, ForeignKey('teams.id'), nullable=False)
    team = relationship(
        'Team', backref=backref('team_events'))

    is_active = Column(Boolean, default=True)
    last_submission_name = Column(String, default=None)
    signup_timestamp = Column(DateTime, nullable=False)
    approved = Column(Boolean, default=False)

    leaderboard_html = Column(String, default=None)
    failed_leaderboard_html = Column(String, default=None)
    new_leaderboard_html = Column(String, default=None)

    UniqueConstraint(event_id, team_id, name='et_constraint')

    def __init__(self, event, team):
        self.event = event
        self.team = team
        self.signup_timestamp = datetime.datetime.utcnow()

    def __repr__(self):
        repr = '{}/{}'.format(self.event, self.team)
        return repr
