import os
import bcrypt
import hashlib
import datetime
import pandas as pd
from collections import OrderedDict
from sqlalchemy.orm import relationship
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.ext.hybrid import hybrid_property
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Enum, \
    DateTime, Boolean, UniqueConstraint
from ..config import submissions_path

from ..config import get_session, get_engine
# so set engine (call config.set_engine_and_session) before importing model
engine = get_engine()
session = get_session()


# These should go in config, later into the ramps table
max_members_per_team = 3  # except for users own team
opening_timestamp = None
public_opening_timestamp = None  # before teams can see only their own scores
closing_timestamp = None

# We will need a ramp_id, user_id, team_id


DBBase = declarative_base()


class User(DBBase):
    __tablename__ = 'users'

    user_id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    hashed_password = Column(String, nullable=False)
    lastname = Column(String, nullable=False)
    firstname = Column(String, nullable=False)
    email = Column(String, nullable=False, unique=True)
    linkedin_url = Column(String, default=None)
    twitter_url = Column(String, default=None)
    facebook_url = Column(String, default=None)
    google_url = Column(String, default=None)
    access_level = Column(Enum('admin', 'user'), default='user')
    signup_timestamp = Column(DateTime, default=datetime.datetime.utcnow)

    admined_teams = relationship('Team', back_populates='admin')  # one-to-many

    def get_teams(self):
        teams = session.query(Team).all()
        for team in teams:
            if self in team.get_members():
                yield team

    def get_n_teams(self):
        return len(self.get_teams())

    def __repr__(self):
        repr = 'User(name={}, lastname={}, firstname={}, email={})'.format(
            self.name, self.lastname, self.firstname, self.email)
        return repr


class Team(DBBase):
    __tablename__ = 'teams'

    team_id = Column(Integer, primary_key=True)
    admin_id = Column(Integer, ForeignKey('users.user_id'))
    # initiator asks for merge, acceptor accepts
    initiator_team_id = Column(
        Integer, ForeignKey('teams.team_id'), default=None)
    acceptor_team_id = Column(
        Integer, ForeignKey('teams.team_id'), default=None)
    name = Column(String, nullable=False, unique=True)
    creation_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    is_active = Column(Boolean, default=True)  # ->ramp_teams

    # one-to-many, ->ramp_teams
    submissions = relationship('Submission', back_populates='team')

    def get_members(self):
        if self.initiator_team_id is not None:
            initiator = session.query(Team).get(self.initiator_team_id)
            # "yield from" in Python 3.3
            for member in initiator.get_members():
                yield member
            acceptor = session.query(Team).get(self.acceptor_team_id)
            for member in acceptor.get_members():
                yield member
        else:
            yield self.admin

    def get_n_members(self):
        return len(list(self.get_members()))

    admin = relationship('User', back_populates='admined_teams')  # many-to-one

    def __repr__(self):
        repr = 'Team(name={}, admin_name={}, size={}, is_active={})'.format(
            self.name, self.admin.name, self.get_n_members(), self.is_active)
        return repr


class Submission(DBBase):
    __tablename__ = 'submissions'

    submission_id = Column(Integer, primary_key=True)
    team_id = Column(Integer, ForeignKey('teams.team_id'), nullable=False)
    name = Column(String, nullable=False)
    file_list = Column(String, nullable=False)
    submission_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    training_timestamp = Column(DateTime)
    scoring_timestamp = Column(DateTime)
    valid_score = Column(Float, default=0.0)  # cv
    test_score = Column(Float, default=0.0)  # holdout
    contributivity = Column(Integer, default=0)
    train_time = Column(Integer, default=0)
    test_time = Column(Integer, default=0)
    trained_state = Column(Enum('new', 'checked', 'trained', 'error', 'scored',
                                'ignore'), default='new')
    tested_state = Column(Enum(
        'new', 'tested', 'scored', 'error'), default='new')
    is_valid = Column(Boolean, default=True)  # user can delete but we keep
    is_to_ensemble = Column(Boolean, default=True)  # we can forget bad models
    notes = Column(String, default='') # eg, why is it disqualified
    team = relationship('Team', back_populates='submissions')  # one-to-many

    UniqueConstraint(team_id, name)  # later also ramp_id

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

    def get_submission_path(self, submissions_path=submissions_path):
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


def get_hashed_password(plain_text_password):
    """Hash a password for the first time
    (Using bcrypt, the salt is saved into the hash itself)"""
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())


def check_password(plain_text_password, hashed_password):
    """Check hased password. Using bcrypt, the salt is saved into the
    hash itself"""
    return bcrypt.checkpw(plain_text_password, hashed_password)


class NameClashError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def create_user(name, password, lastname, firstname, email):
    hashed_password = get_hashed_password(password)
    user = User(name=name, hashed_password=hashed_password,
                lastname=lastname, firstname=firstname, email=email)
    # Creating default team with the same name as the user
    # user is admin of her own team
    team = Team(name=name, admin=user)
    session.add(team)
    session.add(user)
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        message = ''
        try:
            session.query(User).filter_by(name=name).one()
            message += 'username is already in use'
        except NoResultFound:
            # We only check for team names if username is not in db
            try:
                session.query(Team).filter_by(name=name).one()
                message += 'username is already in use as a team name'
            except NoResultFound:
                pass
        try:
            session.query(User).filter_by(email=email).one()
            if len(message) > 0:
                message += ' and '
            message += 'email is already in use'
        except NoResultFound:
            pass
        if len(message) > 0:
            raise NameClashError(message)
        else:
            raise e
    return user


class MergeTeamError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def merge_teams(name, initiator_name, acceptor_name):
    initiator = session.query(Team).filter_by(name=initiator_name).one()
    acceptor = session.query(Team).filter_by(name=acceptor_name).one()
    if not initiator.is_active:
        raise MergeTeamError('Merge initiator is not active')
    if not acceptor.is_active:
        raise MergeTeamError('Merge acceptor is not active')

    # Testing if team size is <= max_members_per_team
    n_members_initiator = initiator.get_n_members()
    n_members_acceptor = acceptor.get_n_members()
    n_members_new = n_members_initiator + n_members_acceptor
    if n_members_new > max_members_per_team:
        raise MergeTeamError(
            'Too big team: new team would be of size {}, the max is {}'.format(
                n_members_new, max_members_per_team))

    members_initiator = initiator.get_members()
    members_acceptor = acceptor.get_members()

    # Testing if team (same members) exists under a different name. If the
    # name is the same, we break. If the loop goes through, we add new team.
    members_set = set(members_initiator).union(set(members_acceptor))
    for team in session.query(Team):
        if members_set == set(team.get_members()):
            if name == team.name:
                break  # ok, but don't add new team, just set them to inactive
            raise MergeTeamError(
                'Team exists with the same members, team name = {}'.format(
                    team.name))
    else:
        team = Team(name=name, admin=initiator.admin,
                    initiator_team_id=initiator.team_id,
                    acceptor_team_id=acceptor.team_id)
        session.add(team)
    initiator.is_active = False
    acceptor.is_active = False
    try:
        session.commit()
    except IntegrityError as e:
        session.rollback()
        try:
            session.query(Team).filter_by(name=name).one()
            raise NameClashError('team name is already in use')
        except NoResultFound:
            raise e
    return team


class DuplicateSubmissionError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def make_submission(team_name, name, file_list):
    team = session.query(Team).filter_by(name=team_name).one()
    submission = session.query(Submission).filter_by(
        name=name, team=team).one_or_none()
    if submission is None:
        submission = Submission(name=name, team=team, file_list=file_list)
        session.add(submission)
    else:
        if submission.trained_state == 'new' or\
                submission.trained_state == 'error' or\
                submission.tested_state == 'error':
            submission.trained_state = 'new'
            submission.tested_state = 'new'
        else:
            raise DuplicateSubmissionError(
                'Submission "{}" of team "{}" exists already'.format(
                    name, team_name))

    # We should copy files here
    session.commit()
    return submission


def get_public_leaderboard():
    table_setup = OrderedDict([
        ('team', Team.name),
        ('submission', Submission.name),
        ('score', Submission.valid_score),
        ('contributivity', Submission.contributivity),
        ('train time', Submission.train_time),
        ('test time', Submission.test_time),
        ('submitted at', Submission.submission_timestamp),
    ])
    table_header = table_setup.keys()
    table_columns = table_setup.values()
    join = session.query(Submission, Team, *table_columns).filter(
        Team.team_id == Submission.team_id)
    submissions = join.filter(Submission.is_public_leaderboard).all()
    # We transpose, get rid of Submission and Team, then retranspose
    df = pd.DataFrame(zip(*zip(*submissions)[2:]), columns=table_header)

    html_params = dict(
        escape=False,
        index=False,
        max_cols=None,
        max_rows=None,
        justify='left',
        classes=['ui', 'blue', 'celled', 'table', 'sortable']
    )

    return df.to_html(**html_params)


def print_users():
    print('***************** List of users ****************')
    for user in session.query(User).order_by(User.user_id):
        print('{} belongs to teams:'.format(user))
        for team in user.get_teams():
            print('\t{}'.format(team))


def print_active_teams():
    print('***************** List of active teams ****************')
    for team in session.query(Team).filter(Team.is_active):
        print('{} members:'.format(team))
        for member in team.get_members():
            print('\t{}'.format(member))


def print_submissions():
    print('***************** List of submissions ****************')
    for submission in session.query(Submission).order_by(
            Submission.submission_id):
        print submission
