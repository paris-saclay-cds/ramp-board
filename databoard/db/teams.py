import datetime
from sqlalchemy.orm import relationship
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm.exc import NoResultFound
from sqlalchemy import Column, Integer, String, ForeignKey, DateTime, Boolean

from databoard.db.model_base import DBBase, NameClashError
from databoard.config import get_session, get_engine
# so set engine (call config.set_engine_and_session) before importing model
engine = get_engine()
session = get_session()


# These should go in config, later into the ramps table
max_members_per_team = 3  # except for users own team


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


def print_active_teams():
    print('***************** List of active teams ****************')
    for team in session.query(Team).filter(Team.is_active):
        print('{} members:'.format(team))
        for member in team.get_members():
            print('\t{}'.format(member))
