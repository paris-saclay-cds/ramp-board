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
    """Event table.

    This table contains all information of a RAMP event.

    Parameters
    ----------
    problem_name : str
        The name of the problem.
    name : str
        The name of the event.
    event_title : str
        The title to give for the event (used in the frontend, can contain
        spaces).
    ramp_sandbox_name : str
        Name of the submission which will be considered the sandbox. It will
        correspond to the key ``sandbox_name`` of the dictionary created with
        :func:`ramp_utils.generate_ramp_config`.
    path_ramp_submissions : str
        Path to the deployment RAMP submissions directory. It will corresponds
        to the key ``ramp_submissions_dir`` of the dictionary created with
        :func:`ramp_utils.generate_ramp_config`.
    session : None or :class:`sqlalchemy.orm.Session`, optional
        The session used to perform some required queries. It is a required
        argument when interacting with the database outside of Flask.

    Attributes
    ----------
    id : int
        ID of the table row.
    name : str
        Event name.
    title : str
        Event title.
    problem_id : int
        The problem ID associated with this event.
    problem : :class:`ramp_database.model.Problem`
        The :class:`ramp_database.model.Problem` instance.
    max_members_per_team : int
        The maximum number of members per team.
    max_n_ensemble : int
        The maximum number of models in the ensemble.
    is_send_trained_mails : bool
        Whether or not to send an email when a model is trained.
    is_public : bool
        Whether or not the event is public.
    is_controled_signup : bool
        Whether or not the sign-up to the event is moderated.
    is_competitive : bool
        Whether or not the challenge is in the competitive phase.
    min_duration_between_submission : int
        The amount of time to wait between two submissions.
    opening_timestamp : datetime
        The date and time of the event opening.
    public_opening_timestamp : datetime
        The date and time of the publicly event opening.
    closing_timestamp : datetime
        The date and time of the event closure.
    official_score_name : str
        The name of the official score used to evaluate the submissions.
    combined_combined_valid_score : float
        The combined public score for all folds.
    combine_combined_test_score : float
        The combined private score for all folds.
    combined_foldwise_valid_score : float
        The combined public scores for each fold.
    combined_foldwise_test_score : float
        The combined public scores for each fold.
    n_submissions : int
        The number of submissions for an event.
    public_leaderboard_html_no_links : str
        The public leaderboard in HTML format with links to the submissions.
    public_leaderboard_html_with_links : str
        The public leaderboard in HTML format.
    private_leaderboard_html : str
        The private leaderboard in HTML.
    failed_leaderboard_html : str
        The leaderboard with the failed submissions.
    new_leaderboard_html : str
        The leaderboard with the new submitted submissions.
    public_competition_leaderboard_html : str
        The public leaderboard of the competition in HTML.
    private_competition_leaderboard_html : str
        The private leaderboard of the competition in HTML.
    path_ramp_kit : str
        The path where the kit are located.
    ramp_sandbox_name : str
        Name of the submission which will be considered the sandbox.
    path_ramp_submissions : str
        Path to the deployment RAMP submissions directory. It will correspond
        to the key `ramp_submissions_dir` of the dictionary created with
        :func:`ramp_utils.generate_ramp_config`.
    score_types : list of :class:`ramp_database.model.EventScoreType`
        A back-reference to the score type used in the event.
    event_admins : list of :class:`ramp_database.model.EventAdmin`
        A back-reference to the admin for the event.
    event_teams: list of :class:`ramp_database.model.EventTeam`
        A back-reference to the teams enrolled in the event.
    cv_folds : list of :class:`ramp_database.model.CVFold`
        A back-reference to the CV folds for the event.
    """
    __tablename__ = 'events'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    title = Column(String, nullable=False)

    problem_id = Column(Integer, ForeignKey('problems.id'), nullable=False)
    problem = relationship('Problem',
                           backref=backref('events',
                                           cascade='all, delete-orphan'))

    max_members_per_team = Column(Integer, default=1)
    # max number of submissions in Caruana's ensemble
    max_n_ensemble = Column(Integer, default=80)
    is_send_trained_mails = Column(Boolean, default=True)
    is_send_submitted_mails = Column(Boolean, default=True)
    is_public = Column(Boolean, default=False)
    is_controled_signup = Column(Boolean, default=True)
    # in competitive events participants can select the submission
    # with which they want to participate in the competition
    is_competitive = Column(Boolean, default=False)

    min_duration_between_submissions = Column(Integer, default=15 * 60)
    opening_timestamp = Column(
        DateTime, default=datetime.datetime(2000, 1, 1, 0, 0, 0))
    # before links to submissions in leaderboard are not alive
    public_opening_timestamp = Column(
        DateTime, default=datetime.datetime(2100, 1, 1, 0, 0, 0))
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
    public_competition_leaderboard_html = Column(String, default=None)
    private_competition_leaderboard_html = Column(String, default=None)

    # big change in the database
    ramp_sandbox_name = Column(String, nullable=False, unique=False,
                               default='starting-kit')
    path_ramp_submissions = Column(String, nullable=False, unique=False)

    def __init__(self, problem_name, name, event_title,
                 ramp_sandbox_name, path_ramp_submissions, session=None):
        self.name = name
        self.ramp_sandbox_name = ramp_sandbox_name
        self.path_ramp_submissions = path_ramp_submissions
        if session is None:
            self.problem = Problem.query.filter_by(name=problem_name).one()
        else:
            self.problem = (session.query(Problem)
                                   .filter(Problem.name == problem_name)
                                   .one())
        self.title = event_title

    def __repr__(self):
        return 'Event({})'.format(self.name)

    def set_n_submissions(self):
        """Set the number of submissions for the current event by checking
        each team."""
        self.n_submissions = 0
        for event_team in self.event_teams:
            # substract one for starting kit
            self.n_submissions += len(event_team.submissions) - 1

    @property
    def Predictions(self):
        """:class:`rampwf.prediction_types.base.BasePrediction`: Predictions
        for the given event."""
        return self.problem.Predictions

    @property
    def workflow(self):
        """:class:`ramp_database.model.Workflow`: The workflow used for the
        event."""
        return self.problem.workflow

    @property
    def official_score_type(self):
        """:class:`ramp_database.model.EventScoreType`: The score type for the
        current event."""
        return (EventScoreType.query
                              .filter_by(event=self,
                                         name=self.official_score_name)
                              .one())

    def get_official_score_type(self, session):
        """Get the type of the default score used for the current event.

        Parameters
        ----------
        session : :class:`sqlalchemy.orm.Session`
            The session used to make the query.
        Returns
        -------
        event_type_score : :class:`ramp_database.model.EventTypeScore`
            The default type score for the current event.
        """
        return (session.query(EventScoreType)
                       .filter(EventScoreType.event == self)
                       .filter(EventScoreType.name == self.official_score_name)
                       .one())

    @property
    def official_score_function(self):
        """callable: The default function used for scoring in the event."""
        return self.official_score_type.score_function

    @property
    def combined_combined_valid_score_str(self):
        """str: Convert to string the combined public score for all folds."""
        return (None if self.combined_combined_valid_score is None
                else str(round(self.combined_combined_valid_score,
                               self.official_score_type.precision)))

    @property
    def combined_combined_test_score_str(self):
        """str: Convert to string the combined private score for all folds."""
        return (None if self.combined_combined_test_score is None
                else str(round(self.combined_combined_test_score,
                               self.official_score_type.precision)))

    @property
    def combined_foldwise_valid_score_str(self):
        """str: Convert to string the combined public score for each fold."""
        return (None if self.combined_foldwise_valid_score is None
                else str(round(self.combined_foldwise_valid_score,
                               self.official_score_type.precision)))

    @property
    def combined_foldwise_test_score_str(self):
        """str: Convert to string the combined public score for each fold."""
        return (None if self.combined_foldwise_test_score is None
                else str(round(self.combined_foldwise_test_score,
                               self.official_score_type.precision)))

    @property
    def is_open(self):
        """bool: Whether or not the event is opened."""
        now = datetime.datetime.utcnow()
        return self.closing_timestamp > now > self.opening_timestamp

    @property
    def is_public_open(self):
        """bool: Whether or not the public phase of the event is opened."""
        now = datetime.datetime.utcnow()
        return self.closing_timestamp > now > self.public_opening_timestamp

    @property
    def is_closed(self):
        """bool: Whether or not the event is closed."""
        now = datetime.datetime.utcnow()
        return now > self.closing_timestamp

    @property
    def n_jobs(self):
        """int: The number of cv fold which can be used as number of jobs."""
        return sum(1 for cv_fold in self.cv_folds if cv_fold.type == 'live')

    @property
    def n_participants(self):
        """int: The number of participants to the event."""
        return len(self.event_teams)


class EventScoreType(Model):
    """EventScoreType table.

    This is a many-to-one relationship between Event and ScoreType. Stores the
    ScoresTypes for each event.
    For each Event / ScoreType combo, also a new record in ScoreType is
    created, which is not that useful (TODO consider removing ScoreType table)

    Parameters
    ----------
    event : :class:`ramp_database.model.Event`
        The event instance.
    score_type_object : :class:`rampwf.score_types`
        A scoring instance.

    Attributes
    ----------
    id : int
        The ID of the table row.
    name : str
        The name of the score.
    event_id : int
        The ID of the event associated.
    event : :class:`ramp_database.model.Event`
        The event instance.
    score_type_id : int
        The ID of the score.
    score_type : :class:`ramp_database.model.ScoreType`
        The score type instance.
    precision : int
        The numerical precision of the score.
    submissions : list of :class:`ramp_database.model.SubmissionScore`
        A back-reference of the submissions for the event/score type.
    """
    __tablename__ = 'event_score_types'

    id = Column(Integer, primary_key=True)
    # Can be renamed, default is the same as score_type.name
    name = Column(String, nullable=False)

    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)
    event = relationship('Event',
                         backref=backref('score_types',
                                         cascade='all, delete-orphan'))

    score_type_id = Column(Integer, ForeignKey('score_types.id'),
                           nullable=False)
    score_type = relationship('ScoreType', backref=backref('events'))

    # display precision in n_digits
    # default is the same as score_type.precision
    precision = Column(Integer)

    UniqueConstraint(event_id, score_type_id, name='es_constraint')
    UniqueConstraint(event_id, name, name='en_constraint')

    def __init__(self, event, score_type_object):
        self.event = event
        self.score_type = ScoreType(str(uuid.uuid4()), True, 0, 1)
        # XXX after migration we should store the index of the
        # score_type so self.score_type_object (should be renamed
        # score_type) wouldn't have to do a search each time.
        self.name = score_type_object.name
        self.precision = score_type_object.precision

    def __repr__(self):
        return '{}: {}'.format(self.name, self.event)

    @property
    def score_type_object(self):
        """:class:`rampwf.score_types`: Score type object."""
        score_types = self.event.problem.module.score_types
        for score_type in score_types:
            if score_type.name == self.name:
                return score_type

    @property
    def score_function(self):
        """callable: Scoring function."""
        return self.score_type_object.score_function

    @property
    def is_lower_the_better(self):
        """bool: Whether a lower score is better."""
        return self.score_type_object.is_lower_the_better

    @property
    def minimum(self):
        """float: the lower bound of the score."""
        return self.score_type_object.minimum

    @property
    def maximum(self):
        """float: the higher bound of the score."""
        return self.score_type_object.maximum

    @property
    def worst(self):
        """float: the worst possible score."""
        return self.score_type_object.worst


class EventAdmin(Model):
    """EventAdmin table.

    This is a many-to-many relationship between Event and User to defined
    admins.

    Parameters
    ----------
    event : :class:`ramp_database.model.Event`
        The event instance.
    admin : :class:`ramp_database.model.User`
        The user instance.

    Attributes
    ----------
    id : int
        The ID of the table row.
    event_id : int
        The ID of the event.
    event : :class:`ramp_database.model.Event`
        The event instance.
    admin_id : int
        The ID of the user defined as an admin.
    admin : :class:`ramp_database.model.User`
        The user instance.
    """
    __tablename__ = 'event_admins'

    id = Column(Integer, primary_key=True)

    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)
    event = relationship('Event',
                         backref=backref('event_admins',
                                         cascade='all, delete-orphan'))

    admin_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    admin = relationship('User',
                         backref=backref('admined_events',
                                         cascade='all, delete-orphan'))


class EventTeam(Model):
    """EventTeam table.

    This is a many-to-many relationship between Event and Team.

    Parameters
    ----------
    event : :class:`ramp_database.model.Event`
        The event instance.
    team : :class:`ramp_database.model.Team`
        The team instance.

    Attributes
    ----------
    id : int
        The ID of a row in the table.
    event_id : int
        The ID of the event.
    event : :class:`ramp_database.model.Event`
        The event instance.
    team_id : int
        The ID of the team.
    team : :class:`ramp_database.model.Team`
        The team instance.
    is_active : bool
        Whether the team is active for the event.
    last_submission_name : str
        The name of the last submission to the event.
    signup_timestamp : datetime
        The date and time when the team signed up for the event.
    approved : bool
        Whether the team has been approved to participate to the event.
    leaderboard_html : str
        The leaderboard for the team for the specific event.
    failed_leaderboard_html : str
        The failed submission board for the team for the specific event.
    new_leaderboard_html : str
        The new submission board for the team for the specific event.
    submissions : list of :class:`ramp_database.model.Submission`
        A back-reference to the submissions associated with this event/team.
    """
    __tablename__ = 'event_teams'

    id = Column(Integer, primary_key=True)

    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)
    event = relationship('Event',
                         backref=backref('event_teams',
                                         cascade='all, delete-orphan'))

    team_id = Column(Integer, ForeignKey('teams.id'), nullable=False)
    team = relationship('Team',
                        backref=backref('team_events',
                                        cascade='all, delete-orphan'))

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
        return '{}/{}'.format(self.event, self.team)
