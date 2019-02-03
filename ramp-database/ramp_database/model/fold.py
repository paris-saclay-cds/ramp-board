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
    """CVFold table.

    Storing train and test folds, more precisely: train and test indices for
    a single fold. Multiple records of this table are then linked to a
    submission (through :class:`ramp_database.model.SubmissionOnCVFold`).

    Attributes
    ----------
    id : int
        The ID of the table row.
    type : {'live', 'test'}
        The type of the CV fold.
    train_is : ndarray
        The training indices.
    test_is : ndarray
        The testing indices.
    event_id : int
        The ID of the event.
    event : :class:`ramp_database.model.Event`
        The event instance.
    submissions : list of :class:`ramp_database.model.SubmissionOnCVFold`
        A back-reference to the submission linked with this fold.
    """

    __tablename__ = 'cv_folds'

    id = Column(Integer, primary_key=True)
    type = Column(cv_fold_types, default='live')

    train_is = Column(NumpyType, nullable=False)
    test_is = Column(NumpyType, nullable=False)

    event_id = Column(Integer, ForeignKey('events.id'), nullable=False)
    event = relationship('Event',
                         backref=backref('cv_folds',
                                         cascade='all, delete-orphan'))

    @staticmethod
    def _pretty_printing(array):
        """Make pretty printing of an array by skipping portion when it is too
        large."""
        if array.size > 10:
            return 'fold {} ... {}'.format(str(array[:5])[:-1],
                                           str(array[-5:])[1:])
        return 'fold {}'.format(array)

    def __repr__(self):
        train_repr = self._pretty_printing(self.train_is)
        test_repr = self._pretty_printing(self.test_is)
        return 'train ' + train_repr + '\n' + ' test ' + test_repr
