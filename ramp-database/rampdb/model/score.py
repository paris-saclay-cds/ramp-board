from sqlalchemy import Float
from sqlalchemy import Column
from sqlalchemy import String
from sqlalchemy import Integer
from sqlalchemy import Boolean

from ramputils import encode_string

from .base import Model

__all__ = ['ScoreType']


# Deprecated: scort types are now defined in problem.py.
# EventScoreType.score_type should be deleted then DB migrated.
class ScoreType(Model):
    __tablename__ = 'score_types'

    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False, unique=True)
    is_lower_the_better = Column(Boolean, nullable=False)
    minimum = Column(Float, nullable=False)
    maximum = Column(Float, nullable=False)

    def __init__(self, name, is_lower_the_better, minimum, maximum):
        self.name = name
        self.is_lower_the_better = is_lower_the_better
        self.minimum = minimum
        self.maximum = maximum

    def __repr__(self):
        return 'ScoreType(name={})'.format(encode_string(self.name))
