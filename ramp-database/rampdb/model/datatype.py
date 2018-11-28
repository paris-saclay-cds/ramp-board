import zlib
import numpy as np
import pickle

from sqlalchemy import LargeBinary
from sqlalchemy import TypeDecorator

__all__ = ['NumpyType']


class NumpyType(TypeDecorator):
    """Storing zipped numpy arrays."""
    impl = LargeBinary

    def process_bind_param(self, value, dialect):
        # we convert the initial value into np.array to handle None and lists
        return zlib.compress(np.array(value).dumps())

    def process_result_value(self, value, dialect):
        return pickle.loads(zlib.decompress(value))
