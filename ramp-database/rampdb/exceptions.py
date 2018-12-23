"""
The :mod:`rampdb.exceptions` module include all custom warnings and errors
used in ``ramp-database``.
"""

__all__ = ['UnknownStateError']


class UnknownStateError(Exception):
    pass
