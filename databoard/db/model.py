import os
import hashlib
import datetime
from flask import Flask
from flask.ext.sqlalchemy import SQLAlchemy
from sqlalchemy.ext.hybrid import hybrid_property
import databoard.config as config

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + config.db_f_name
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
db = SQLAlchemy(app)

max_members_per_team = 3  # except for users own team
opening_timestamp = None
public_opening_timestamp = None  # before teams can see only their own scores
closing_timestamp = None


class User(db.Model):
    __tablename__ = 'users'

    user_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False, unique=True)
    hashed_password = db.Column(db.String, nullable=False)
    lastname = db.Column(db.String, nullable=False)
    firstname = db.Column(db.String, nullable=False)
    email = db.Column(db.String, nullable=False, unique=True)
    linkedin_url = db.Column(db.String, default=None)
    twitter_url = db.Column(db.String, default=None)
    facebook_url = db.Column(db.String, default=None)
    google_url = db.Column(db.String, default=None)
    access_level = db.Column(db.Enum('admin', 'user'), default='user')
    signup_timestamp = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    is_validated = db.Column(db.Boolean, default=False)  # admin has to valid

    #admined_teams = db.relationship(
    #    'Team', back_populates='admin')  # one-to-many

    def __repr__(self):
        repr = 'User(name={}, lastname={}, firstname={}, email={}, admined_teams={})'.format(
            self.name, self.lastname, self.firstname, self.email, self.admined_teams)
        return repr


class Team(db.Model):
    __tablename__ = 'teams'

    team_id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String, nullable=False, unique=True)

    admin_id = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    admin = db.relationship('User', backref=db.backref('admined_teams'))

    # initiator asks for merge, acceptor accepts
    initiator_id = db.Column(
        db.Integer, db.ForeignKey('teams.team_id'), default=None)
    initiator = db.relationship(
        'Team', primaryjoin=('Team.initiator_id == Team.team_id'), uselist=False)

    acceptor_id = db.Column(
        db.Integer, db.ForeignKey('teams.team_id'), default=None)
    acceptor = db.relationship(
        'Team', primaryjoin=('Team.acceptor_id == Team.team_id'), uselist=False)

    creation_timestamp = db.Column(
        db.DateTime, default=datetime.datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)  # ->ramp_teams

    def __repr__(self):
        repr = 'Team(name={}, admin_name={}, is_active={}, initiator={}, acceptor={})'.format(
            self.name, self.admin.name, self.is_active, self.initiator, self.acceptor)
        return repr


class Submission(db.Model):
    __tablename__ = 'submissions'

    submission_id = db.Column(db.Integer, primary_key=True)
    team_id = db.Column(
        db.Integer, db.ForeignKey('teams.team_id'), nullable=False)
    # one-to-many, ->ramp_teams
    team = db.relationship('Team', backref=db.backref('submissions'))
    name = db.Column(db.String, nullable=False)
    file_list = db.Column(db.String, nullable=False)
    submission_timestamp = db.Column(
        db.DateTime, default=datetime.datetime.utcnow)
    training_timestamp = db.Column(db.DateTime)
    scoring_timestamp = db.Column(db.DateTime)
    valid_score = db.Column(db.Float, default=0.0)  # cv
    test_score = db.Column(db.Float, default=0.0)  # holdout
    contributivity = db.Column(db.Integer, default=0)
    train_time = db.Column(db.Integer, default=0)
    test_time = db.Column(db.Integer, default=0)
    trained_state = db.Column(db.Enum(
        'new', 'checked', 'trained', 'error', 'scored', 'ignore'),
        default='new')
    tested_state = db.Column(db.Enum(
        'new', 'tested', 'scored', 'error'), default='new')
    is_valid = db.Column(
        db.Boolean, default=True)  # user can delete but we keep
    is_to_ensemble = db.Column(
        db.Boolean, default=True)  # we can forget bad models
    notes = db.Column(db.String, default='')  # eg, why is it disqualified

    db.UniqueConstraint(team_id, name)  # later also ramp_id

    @hybrid_property
    def is_public_leaderboard(self):
        return self.is_valid and self.trained_state == 'scored'

    @hybrid_property
    def is_private_leaderboard(self):
        return self.is_valid and self.tested_state == 'scored'

    def _get_submission_hash(self):
        sha_hasher = hashlib.sha1()
        sha_hasher.update(self.team.name)
        sha_hasher.update(self.name)
        model_hash = 'm{}'.format(sha_hasher.hexdigest())
        return model_hash

    def get_submission_path(self, submissions_path=config.submissions_path):
        submission_hash = self._get_submission_hash()
        team_path = os.path.join(submissions_path, self.team.name)
        submission_path = os.path.join(team_path, submission_hash)
        return team_path, submission_path

    def __repr__(self):
        repr = 'Submission(team_name={}, name={}, file_list={}, '\
            'trained_state={}, tested_state={})'.format(
                self.team.name, self.name, self.file_list, self.trained_state,
                self.tested_state)
        return repr


class NameClashError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class MergeTeamError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class DuplicateSubmissionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

