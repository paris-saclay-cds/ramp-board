import datetime

from sqlalchemy import Enum
from sqlalchemy import Float
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship

from .base import Model
from .event import EventTeam

__all__ = [
    'User',
    'UserInteraction',
]


class User(Model):
    """User table.

    Parameters
    ----------
    name : str
        The user name.
    hashed_password : str
        The hashed password.
    lastname : str
        The user's last name.
    firstname : str
        The user's first name.
    email : str
        The user's email
    access_level : {'admin', 'user', 'asked'}
        The user's admin level.
    hidden_notes : str
        Some hidden notes.
    signup_timestamp : datetime
        The date and time of the user's sign-up.
    linkedin_url : str
        The user's LinkedIn URL.
    twitter_url : str
        The user's Twitter URL.
    facebook_url : str
        The user's Facebook URL.
    google_url : str
        The user's Google URL.
    github_url : str
        The user's GitHub URL.
    website_url : str
        The user's personal website URL.
    bio : str
        The user's biography.
    is_want_news : bool
        Whether or not the user wants to receive RAMP news.

    Attributes
    ----------
    id : int
        The ID of the table row.
    name : str
        The user name.
    hashed_password : str
        The hashed password.
    lastname : str
        The user's last name.
    firstname : str
        The user's first name.
    email : str
        The user's email
    access_level : {'admin', 'user', 'asked', 'not_confirmed'}
        The user's admin level. The possible access level are:

        * 'admin' : RAMP administrator;
        * 'user': RAMP user;
        * 'asked': asked to be a RAMP user and confirmed via emails;
        * 'not_confirmed': signed-up through the frontend but did not confirm
          yet by email.
    hidden_notes : str
        Some hidden notes.
    signup_timestamp : datetime
        The date and time of the user's sign-up.
    linkedin_url : str
        The user's LinkedIn URL.
    twitter_url : str
        The user's Twitter URL.
    facebook_url : str
        The user's Facebook URL.
    google_url : str
        The user's Google URL.
    github_url : str
        The user's GitHub URL.
    website_url : str
        The user's personal website URL.
    bio : str
        The user's biography.
    is_want_news : bool
        Whether or not the user wants to receive RAMP news.
    is_authenticated : bool
        Whether or not the user logged-in.
    admined_events : list of :class:`ramp_database.model.EventAdmin`
        A back-reference to the events administrated by the user.
    submission_similaritys : list of \
:class:`ramp_database.model.SubmissionSimilarity`
        A back-reference to the submission similarity.
    admined_teams : list of :class:`ramp_database.model.Team`
        A back-reference to the teams administrated by the user.
    """
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    name = Column(String(20), nullable=False, unique=True)
    hashed_password = Column(String, nullable=False)
    lastname = Column(String(256), nullable=False)
    firstname = Column(String(256), nullable=False)
    email = Column(String(256), nullable=False, unique=True)
    linkedin_url = Column(String(256), default=None)
    twitter_url = Column(String(256), default=None)
    facebook_url = Column(String(256), default=None)
    google_url = Column(String(256), default=None)
    github_url = Column(String(256), default=None)
    website_url = Column(String(256), default=None)
    hidden_notes = Column(String, default=None)
    bio = Column(String(1024), default=None)
    is_want_news = Column(Boolean, default=True)
    access_level = Column(
        Enum('admin', 'user', 'asked', 'not_confirmed', name='access_level'),
        default='asked'
    )
    signup_timestamp = Column(DateTime, nullable=False)

    # Flask-Login fields
    is_authenticated = Column(Boolean, default=False)

    def __init__(self, name, hashed_password, lastname, firstname, email,
                 access_level='user', hidden_notes='', linkedin_url='',
                 twitter_url='', facebook_url='', google_url='', github_url='',
                 website_url='', bio='', is_want_news=True):
        self.name = name
        self.hashed_password = hashed_password
        self.lastname = lastname
        self.firstname = firstname
        self.email = email
        self.access_level = access_level
        self.hidden_notes = hidden_notes
        self.signup_timestamp = datetime.datetime.utcnow()
        self.linkedin_url = linkedin_url
        self.twitter_url = twitter_url
        self.facebook_url = facebook_url
        self.google_url = google_url
        self.github_url = github_url
        self.website_url = website_url
        self.bio = bio
        self.is_want_news = is_want_news

    @property
    def is_active(self):
        """bool: Return True."""
        return True

    @property
    def is_anonymous(self):
        """bool: Return False."""
        return False

    def get_id(self):
        """str: Return the user ID."""
        return str(self.id)

    def __str__(self):
        return 'User({})'.format(self.name)

    def __repr__(self):
        return ("User(name={}, lastname={}, firstname={}, email={}, "
                "admined_teams={})"
                .format(self.name, self.lastname, self.firstname,
                        self.email, self.admined_teams))


user_interaction_type = Enum(
    'copy',
    'download',
    'giving credit',
    'landing',
    'login',
    'logout',
    'looking at error',
    'looking at event',
    'looking at problem',
    'looking at problems',
    'looking at leaderboard',
    'looking at my_submissions',
    'looking at private leaderboard',
    'looking at submission',
    'looking at user',
    'save',
    'signing up at event',
    'submit',
    'upload',
    name='user_interaction_type'
)


# TODO: the ip should not be read from the environment variable but from the
# config file instead.
class UserInteraction(Model):
    """UserInteraction table.

    This class is used to record the interaction of a user with the frontend.

    Parameters
    ----------
    interactions : None or str, default is None
        The type of interaction.
    user : None or :class:`ramp_database.model.User`, default is None
        The user instance.
    problem : None or :class:`ramp_database.model.Problem`, default is None
        The problem instance.
    event : None or :class:`ramp_database.model.Event`, default is None
        The event instance.
    ip : None or str, default is None
        The ip address from the server.
    note : None or str, default is None
        Some notes.
    submission : None or :class:`ramp_database.model.Submission`, \
default is None
        The submission instance.
    submission_file : None or :class:`ramp_database.model.SubmissionFile`, \
default is None
        The submission file instance.
    diff : None or str, default is None
        The difference between two submissions.
    similarity : None or float, default is None
        The similarity of the submission.

    Attributes
    ----------
    id : int
        The ID of the table row.
    timestamp : datetime
        The date and time the interaction was created.
    interactions : str
        The type of interaction.
    note : str
        Some note regarding the interaction.
    submission_file_diff : str
        The difference between two submission files.
    submission_file_similarity : float
        The similarity between two submission files.
    ip : str
        The IP of the remove server.
    user_id : int
        The ID of the user linked to the interaction.
    user : :class:`ramp_database.model.User`
        The user instance.
    problem_id : int
        The ID of the problem.
    problem : :class:`ramp_database.model.Problem`
        The problem instance.
    event_team_id : int
        The ID of the event/team.
    event_team : :class:`ramp_database.model.EventTeam`
        The event/team instance.
    submission_id : int
        The submission ID.
    submission : :class:`ramp_database.model.Submission`
        The submission instance.
    submission_file_id : int
        The submission file ID.
    submission_file : :class:`ramp_database.model.SubmissionFile`
        The submission file instance.
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    """
    __tablename__ = 'user_interactions'

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False)
    interaction = Column(user_interaction_type, nullable=False)
    note = Column(String, default=None)
    submission_file_diff = Column(String, default=None)
    submission_file_similarity = Column(Float, default=None)
    ip = Column(String, default=None)

    user_id = Column(
        Integer, ForeignKey('users.id'))
    user = relationship('User',
                        backref=backref('user_interactions',
                                        cascade='all, delete-orphan'))

    problem_id = Column(
        Integer, ForeignKey('problems.id'))
    problem = relationship('Problem', backref=backref(
        'user_interactions', cascade='all, delete-orphan'))

    event_team_id = Column(
        Integer, ForeignKey('event_teams.id'))
    event_team = relationship('EventTeam', backref=backref(
        'user_interactions', cascade='all, delete-orphan'))

    submission_id = Column(
        Integer, ForeignKey('submissions.id'))
    submission = relationship('Submission', backref=backref(
        'user_interactions', cascade='all, delete-orphan'))

    submission_file_id = Column(
        Integer, ForeignKey('submission_files.id'))
    submission_file = relationship('SubmissionFile', backref=backref(
        'user_interactions', cascade='all, delete-orphan'))

    def __init__(self, interaction=None, user=None, problem=None, event=None,
                 ip=None, note=None, submission=None, submission_file=None,
                 diff=None, similarity=None, session=None):
        self.timestamp = datetime.datetime.utcnow()
        self.interaction = interaction
        self.user = user
        self.problem = problem
        if event is not None and user is not None:
            # There should always be an active user team, if not, throw an
            # exception
            # The current code works only if each user admins a single team.
            if session is None:
                self.event_team = EventTeam.query.filter_by(
                    event=event, team=user.admined_teams[0]).one_or_none()
            else:
                self.event_team = \
                    (session.query(EventTeam)
                            .filter(EventTeam.event == event)
                            .filter(EventTeam.team == user.admined_teams[0])
                            .one_or_none())
        self.ip = ip
        self.note = note
        self.submission = submission
        self.submission_file = submission_file
        self.submission_file_diff = diff
        self.submission_file_similarity = similarity

# The following function was implemented to handle user interaction dump
# but it turned out that the db insertion was not the CPU sink. Keep it
# for a while if the site is still slow.

    # def __init__(self, line=None, interaction=None, user=None, event=None,
    #              ip=None, note=None, submission=None, submission_file=None,
    #              diff=None, similarity=None):
    #     if line is None:
    #         # normal real-time construction using kwargs
    #         self.timestamp = datetime.datetime.utcnow()
    #         self.interaction = interaction
    #         self.user = user
    #         if event is not None:
    #             self.event_team = get_active_user_event_team(event, user)
    #         if ip is None:
    #             self.ip = request.environ['REMOTE_ADDR']
    #         else:
    #             self.ip = ip
    #         self.note = note
    #         self.submission = submission
    #         self.submission_file = submission_file
    #         self.submission_file_diff = diff
    #         self.submission_file_similarity = similarity
    #     else:
    #         # off-line construction using dump from
    #         # config.user_interactions_f_name
    #         tokens = line.split(';')
    #         self.timestamp = eval(tokens[0])
    #         self.interaction = eval(tokens[1])
    #         self.note = eval(tokens[2])
    #         self.submission_file_diff = eval(tokens[3])
    #         self.submission_file_similarity = eval(tokens[4])
    #         self.ip = eval(tokens[5])
    #         self.user_id = eval(tokens[6])
    #         self.event_team_id = eval(tokens[7])
    #         self.submission_id = eval(tokens[8])
    #         self.submission_file_id = eval(tokens[9])

    def __repr__(self):
        return "; ".join(repr(member)
                         for member in (self.timestamp, self.interaction,
                                        self.note, self.submission_file_diff,
                                        self.submission_file_similarity,
                                        self.ip, self.user, self.problem,
                                        self.event_team, self.submission,
                                        self.submission_file))

    @property
    def event(self):
        """:class:`ramp_database.model.Event`: The event instance."""
        return self.event_team.event if self.event_team else None

    @property
    def team(self):
        """:class:`ramp_database.model.Team`: The team instance."""
        return self.event_team.team if self.event_team else None
