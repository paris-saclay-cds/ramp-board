import bcrypt
import datetime
from sqlalchemy import create_engine
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, Float, ForeignKey, Enum, \
    DateTime, Boolean, UniqueConstraint


engine = create_engine('sqlite:///:memory:', echo=False)
Session = sessionmaker(bind=engine)
session = Session()
DBBase = declarative_base()

# These should go in config, later into the ramps table
max_members_per_team = 3  # except for users own team
opening_timestamp = None
public_opening_timestamp = None  # before teams can see only their own scores
closing_timestamp = None

# We will allow one team change per ramp
# We will need a ramp_id, user_id, team_id


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
        user_teams = []
        for team in teams:
            if self in team.get_members():
                user_teams.append(team)
        return user_teams

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
    submissions = relationship('Submission', backref="team")

    def get_members(self):
        members = []
        if self.initiator_team_id is not None:
            initiator = session.query(Team).get(self.initiator_team_id)
            members.extend(initiator.get_members())
            acceptor = session.query(Team).get(self.acceptor_team_id)
            members.extend(acceptor.get_members())
        else:
            members.append(self.admin)
        return members

    def get_n_members(self):
        return len(self.get_members())

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
    path = Column(String, nullable=False)
    submission_timestamp = Column(DateTime, default=datetime.datetime.utcnow)
    training_timestamp = Column(DateTime)
    scoring_timestamp = Column(DateTime)
    valid_score = Column(Float, default=0.0)  # cv
    test_score = Column(Float, default=0.0)  # holdout
    contributivity = Column(Integer, default=0)
    train_time = Column(Integer, default=0)
    test_time = Column(Integer, default=0)
    trained_state = Column(Enum(
        'new', 'trained', 'tested', 'error', 'ignore'), default='new')
    scored_state = Column(Enum(
        'train_scored', 'test_scored'), default=None)
    is_valid = Column(Boolean, default=True)  # user can delete but we keep
    is_to_ensemble = Column(Boolean, default=True)  # we can forget bad models


DBBase.metadata.create_all(engine)


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

    n_members_initiator = initiator.get_n_members()
    n_members_acceptor = acceptor.get_n_members()
    n_members_new = n_members_initiator + n_members_acceptor
    if n_members_new > max_members_per_team:
        raise MergeTeamError(
            'Too big team: new team would be of size {}, the max is {}'.format(
                n_members_new, max_members_per_team))
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
