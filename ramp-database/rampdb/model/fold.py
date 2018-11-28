from sqlalchemy import Enum
from sqlalchemy import Column
from sqlalchemy import Integer
from sqlalchemy import ForeignKey
from sqlalchemy.orm import backref
from sqlalchemy.orm import relationship

from .base import Model
from .datatype import NumpyType

__all__ = [
    'CVFold',
]


cv_fold_types = Enum('live', 'test', name='cv_fold_types')


class CVFold(Model):
    """Storing train and test folds, more precisely: train and test indices.

    Created when the ramp event is set up.
    """

    __tablename__ = 'cv_folds'

    id = Column(Integer, primary_key=True)
    type = Column(cv_fold_types, default='live')

    train_is = Column(NumpyType, nullable=False)
    test_is = Column(NumpyType, nullable=False)

    event_id = Column(
        Integer, ForeignKey('events.id'), nullable=False)
    event = relationship('Event', backref=backref(
        'cv_folds', cascade='all, delete-orphan'))

    def __repr__(self):
        return 'fold {}'.format(self.train_is)[:15]
