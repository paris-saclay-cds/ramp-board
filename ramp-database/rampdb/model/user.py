import os
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

from ramputils import encode_string

from .base import Model
from .event import EventTeam

__all__ = [
    'User',
    'UserInteraction',
]


class User(Model):
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
    access_level = Column(Enum(
        'admin', 'user', 'asked', name='access_level'), default='asked')
    # 'asked' needs approval
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
        return True

    @property
    def is_anonymous(self):
        return False

    def get_id(self):
        try:
            return unicode(self.id)  # python 2
        except NameError:
            return str(self.id)  # python 3

    def __str__(self):
        return 'User({})'.format(encode_string(self.name))

    def __repr__(self):
        text = ("User(name={}, lastname={}, firstname={}, email={}, "
                "admined_teams={})"
                .format(
                    encode_string(self.name),
                    encode_string(self.lastname),
                    encode_string(self.firstname),
                    encode_string(self.email),
                    self.admined_teams))
        return text


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


class UserInteraction(Model):
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
    user = relationship('User', backref=backref('user_interactions'))

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
                 diff=None, similarity=None):
        self.timestamp = datetime.datetime.utcnow()
        self.interaction = interaction
        self.user = user
        self.problem = problem
        if event is not None and user is not None:
            # There should always be an active user team, if not, throw an
            # exception
            # The current code works only if each user admins a single team.
            self.event_team = EventTeam.query.filter_by(
                event=event, team=user.admined_teams[0]).one_or_none()
        if ip is None:
            self.ip = os.getenv('REMOTE_ADDR')
        else:
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
        repr = self.timestamp.__repr__()
        repr += ';' + self.interaction.__repr__()
        repr += ';' + self.note.__repr__()
        repr += ';' + self.submission_file_diff.__repr__()
        repr += ';' + self.submission_file_similarity.__repr__()
        repr += ';' + self.ip.__repr__()
        if self.user is None:
            repr += ';None'
        else:
            repr += ';' + self.user.id.__repr__()
        if self.problem is None:
            repr += ';None'
        else:
            repr += ';' + self.problem.id.__repr__()
        if self.event_team is None:
            repr += ';None'
        else:
            repr += ';' + self.event_team.id.__repr__()
        if self.submission is None:
            repr += ';None'
        else:
            repr += ';' + self.submission.id.__repr__()
        if self.submission_file is None:
            repr += ';None'
        else:
            repr += ';' + self.submission_file.id.__repr__()
        return repr

    # @property
    # def submission_file_diff_link(self):
    #     if self.submission_file_diff is None:
    #         return None
    #     return os.path.join(
    #         deployment_path,
    #         ramp_config['submissions_dir'],
    #         'diff_bef24208a45043059',
    #         str(self.id))

    @property
    def event(self):
        if self.event_team:
            return self.event_team.event
        else:
            return None

    @property
    def team(self):
        if self.event_team:
            return self.event_team.team
        else:
            return None
