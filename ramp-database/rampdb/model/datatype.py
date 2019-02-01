import pickle
import zlib

import numpy as np

from sqlalchemy import LargeBinary
from sqlalchemy import TypeDecorator

__all__ = ['NumpyType']


class NumpyType(TypeDecorator):
    """Storing zipped numpy arrays."""
    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        """Compress and pickle NumPy arrays on disk.

        Parameters
        ----------
        value : ndarray
            The array to pickle.
        dialect : Dialect
            Dialect in use.
        Returns
        -------
        pickle : pickle
            The compress pickle.
        """
        # we convert the initial value into np.array to handle None and lists
        return zlib.compress(np.array(value).dumps())

    def process_result_value(self, value, dialect):
        """Load pickle into NumPy array.

        Parameters
        ----------
        value : pickle
            The NumPy array which was dumped into a pickle.
        dialect : Dialect
            Dialect in use.
        Returns
        -------
        array : ndarray
            The NumPy array which has been loaded.
        """
        return pickle.loads(zlib.decompress(value))
