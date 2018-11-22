import datetime

from sqlalchemy import Enum
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import Boolean
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship

from .base import Model
from .event import Event
from .event import EventTeam

__all__ = ['Team']


class Team(Model):
    __tablename__ = 'teams'

    id = Column(Integer, primary_key=True)
    name = Column(String(20), nullable=False, unique=True)

    admin_id = Column(Integer, ForeignKey('users.id'))
    admin = relationship('User', backref=backref('admined_teams'))

    # initiator asks for merge, acceptor accepts
    initiator_id = Column(
        Integer, ForeignKey('teams.id'), default=None)
    initiator = relationship(
        'Team', primaryjoin=('Team.initiator_id == Team.id'), uselist=False)

    acceptor_id = Column(
        Integer, ForeignKey('teams.id'), default=None)
    acceptor = relationship(
        'Team', primaryjoin=('Team.acceptor_id == Team.id'), uselist=False)

    creation_timestamp = Column(DateTime, nullable=False)

    def __init__(self, name, admin, initiator=None, acceptor=None):
        self.name = name
        self.admin = admin
        self.initiator = initiator
        self.acceptor = acceptor
        self.creation_timestamp = datetime.datetime.utcnow()

    def __str__(self):
        str_ = 'Team({})'.format(self.name.encode('utf-8'))
        return str_

    def __repr__(self):
        repr = ("Team(name={}, admin_name={}, initiator={}, acceptor={})"
                .format(self.name.encode('utf-8'),
                        self.admin.name.encode('utf-8'),
                        self.initiator,
                        self.acceptor))
        return repr
