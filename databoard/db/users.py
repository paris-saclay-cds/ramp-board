import bcrypt
import datetime
from sqlalchemy.orm import relationship
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import Column, Integer, String, Enum, DateTime

from databoard.db.model_base import DBBase, NameClashError
from databoard.db.teams import Team
from databoard.config import get_session, get_engine
# so set engine (call config.set_engine_and_session) before importing model
engine = get_engine()
session = get_session()


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


def get_hashed_password(plain_text_password):
    """Hash a password for the first time
    (Using bcrypt, the salt is saved into the hash itself)"""
    return bcrypt.hashpw(plain_text_password, bcrypt.gensalt())


def check_password(plain_text_password, hashed_password):
    """Check hased password. Using bcrypt, the salt is saved into the
    hash itself"""
    return bcrypt.checkpw(plain_text_password, hashed_password)


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


def print_users():
    print('***************** List of users ****************')
    for user in session.query(User).order_by(User.user_id):
        print('{} belongs to teams:'.format(user))
        for team in user.get_teams():
            print('\t{}'.format(team))
