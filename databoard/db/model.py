import os
import string
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

    id_ = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False, unique=True)
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

    id_ = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(20), nullable=False, unique=True)

    admin_id = db.Column(db.Integer, db.ForeignKey('users.id_'))
    admin = db.relationship('User', backref=db.backref('admined_teams'))

    # initiator asks for merge, acceptor accepts
    initiator_id = db.Column(
        db.Integer, db.ForeignKey('teams.id_'), default=None)
    initiator = db.relationship(
        'Team', primaryjoin=('Team.initiator_id == Team.id_'), uselist=False)

    acceptor_id = db.Column(
        db.Integer, db.ForeignKey('teams.id_'), default=None)
    acceptor = db.relationship(
        'Team', primaryjoin=('Team.acceptor_id == Team.id_'), uselist=False)

    creation_timestamp = db.Column(
        db.DateTime, default=datetime.datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)  # ->ramp_teams

    def __repr__(self):
        repr = 'Team(name={}, admin_name={}, is_active={}, initiator={}, acceptor={})'.format(
            self.name, self.admin.name, self.is_active, self.initiator, self.acceptor)
        return repr


# TODO: we shuold have a SubmissionFileType table, describing the type
# of files we are expecting for a given RAMP. Fast unit test should be set up 
# there, and each file should be unit tested right after submission.
# We should have a max_size attribute that we could set when setting up ramps.
# Kozmetics: erhaps mark which file the leaderboard link should point to (right
# now it is set to the first file in the list which is arbitrary).
# We will also have to handle auxiliary files (like csvs or other classes).
# User interface could have a sinlge submission form with a menu containing 
# the file names for a given ramp + an "other" field when users will have to 
# name their files
class SubmissionFile(db.Model):
    __tablename__ = 'submission_files'

    id_ = db.Column(db.Integer, primary_key=True)
    submission_id = db.Column(
        db.Integer, db.ForeignKey('submissions.id_'), nullable=False)
    submission = db.relationship(
        'Submission', backref=db.backref('submission_files'))
    name = db.Column(db.String, nullable=False)

    db.UniqueConstraint(submission_id, name)

    @hybrid_property
    def relative_path(self):
        return self.submission.relative_path + os.path.sep + self.name

    def __repr__(self):
        return 'SubmissionFile(name={}, relative_path={})'.format(
            self.name, self.relative_path)


class Submission(db.Model):
    __tablename__ = 'submissions'

    id_ = db.Column(db.Integer, primary_key=True)
    team_id = db.Column(db.Integer, db.ForeignKey('teams.id_'), nullable=False)
    # one-to-many, ->ramp_teams
    team = db.relationship('Team', backref=db.backref('submissions'))
    name = db.Column(db.String(20), nullable=False)
    hash_ = db.Column(db.String, nullable=False)
    submission_timestamp = db.Column(
        db.DateTime, default=datetime.datetime.utcnow)
    training_timestamp = db.Column(db.DateTime)
    scoring_timestamp = db.Column(db.DateTime)
    valid_score = db.Column(db.Float, default=0.0)  # cv
    test_score = db.Column(db.Float, default=0.0)  # holdout
    contributivity = db.Column(db.Integer, default=0)
    train_time = db.Column(db.Integer, default=0)
    test_time = db.Column(db.Integer, default=0)
    # evaluate right after train/test, so no need for 'scored' states
    state = db.Column(db.Enum('new', 'checked', 'trained', 'tested',
        'train_scored', 'test_scored', 'check_error', 'train_error', 
        'test_error', 'unit_test_error', 'ignore'),
        default='new')
    is_valid = db.Column(
        db.Boolean, default=True)  # user can delete but we keep
    is_to_ensemble = db.Column(
        db.Boolean, default=True)  # we can forget bad models
    notes = db.Column(db.String, default='')  # eg, why is it disqualified

    db.UniqueConstraint(team_id, name)  # later also ramp_id

    def __init__(self, name, team):
        self.name = name
        self.team = team
        sha_hasher = hashlib.sha1()
        sha_hasher.update(self.team.name)
        sha_hasher.update(self.name)
        # We considered using the id, but then it will be given away in the
        # url which is maybe not a good idea.
        self.hash_ = 'm{}'.format(sha_hasher.hexdigest())

    @hybrid_property
    def is_public_leaderboard(self):
        return self.is_valid and self.state == 'train_scored'

    @property
    def relative_path(self):
        hash_ = self.hash_
        relative_path = os.path.join(
            config.submissions_d_name, self.team.name, hash_)
        return relative_path

    @property
    def submission_f_names(self):
        return [submission_file.name for submission_file
                in self.submission_files]

    def get_paths(self, submissions_path=config.submissions_path):
        team_path = os.path.join(submissions_path, self.team.name)
        submission_path = os.path.join(team_path, self.hash_)
        return team_path, submission_path

    @property
    def name_with_link(self):
        return '<a href="' + self.submission_files[0].relative_path + '">' +\
            self.name + '</a>'

    def __repr__(self):
        repr = 'Submission(team_name={}, name={}, submission_files={}, '\
            'state={}, train_time={})'.format(
                self.team.name, self.name, self.submission_files,
                self.state, self.train_time)
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

