from importlib import import_module

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

from .base import Model

__all__ = ['ScoreType']


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
        # to check if the module and all required fields are there
        # self.module
        # self.score_function
        # self.precision

    def __repr__(self):
        repr = 'ScoreType(name={})'.format(self.name)
        return repr

    @property
    def module(self):
        return import_module('.' + self.name, 'databoard.specific.score_types')

    @property
    def score_function(self):
        return self.module.score_function

    @property
    def worst(self):
        if self.is_lower_the_better:
            return self.maximum
        else:
            return self.minimum

    # default display precision in n_digits
    @property
    def precision(self):
        return self.module.precision
