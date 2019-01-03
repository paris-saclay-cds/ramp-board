import datetime

from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import DateTime
from sqlalchemy import ForeignKey
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship

from ramputils import encode_string

from .base import Model

__all__ = ['Team']


class Team(Model):
    """Team table.

    Parameters
    ----------
    name : str
        The name of the team.
    admin : :class:`rampdb.model.User`
        The admin user of the team.
    initiator : None or :class:`rampdb.model.Team`, default is None
        The team initiating a merging.
    acceptor : None or :class:`rampdb.model.Team`, default is None
        The team accepting a merging.

    Attributes
    ----------
    id : int
        The ID of the table row.
    name : str
        The name of the team.
    admin_id : int
        The ID of the admin user.
    admin : :class:`rampdb.model.User`
        The admin user instance.
    initiator_id : int
        The ID of the team asking for merging.
    initiator : :class:`rampdb.model.Team`
        The team instance asking for merging.
    acceptor_id : int
        The ID of the team accepting the merging.
    acceptor : :class:`rampdb.model.Team`
        The team instance accepting the merging.
    team_events : :class:`rampdb.model.EventTeam`
        A back-reference to the events to which the team is enroll.
    """
    __tablename__ = 'teams'

    id = Column(Integer, primary_key=True)
    name = Column(String(20), nullable=False, unique=True)

    admin_id = Column(Integer, ForeignKey('users.id'))
    admin = relationship('User', backref=backref('admined_teams'))

    # initiator asks for merge, acceptor accepts
    initiator_id = Column(Integer, ForeignKey('teams.id'), default=None)
    initiator = relationship(
        'Team', primaryjoin=('Team.initiator_id == Team.id'), uselist=False
    )

    acceptor_id = Column(Integer, ForeignKey('teams.id'), default=None)
    acceptor = relationship(
        'Team', primaryjoin=('Team.acceptor_id == Team.id'), uselist=False
    )

    creation_timestamp = Column(DateTime, nullable=False)

    def __init__(self, name, admin, initiator=None, acceptor=None):
        self.name = name
        self.admin = admin
        self.initiator = initiator
        self.acceptor = acceptor
        self.creation_timestamp = datetime.datetime.utcnow()

    def __str__(self):
        return 'Team({})'.format(encode_string(self.name))

    def __repr__(self):
        return ('Team(name={}, admin_name={}, initiator={}, acceptor={})'
                .format(encode_string(self.name),
                        encode_string(self.admin.name),
                        self.initiator, self.acceptor))
