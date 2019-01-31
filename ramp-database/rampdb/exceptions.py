"""
The :mod:`rampdb.exceptions` module include all custom warnings and errors
used in ``ramp-database``.
"""

__all__ = [
    'DuplicateSubmissionError',
    'MergeTeamError',
    'MissingExtensionError',
    'MissingSubmissionFileError',
    'NameClashError',
    'TooEarlySubmissionError',
    'UnknownStateError'
    ]


class DuplicateSubmissionError(Exception):
    pass


class MergeTeamError(Exception):
    pass


class MissingSubmissionFileError(Exception):
    pass


class MissingExtensionError(Exception):
    pass


class NameClashError(Exception):
    pass


class TooEarlySubmissionError(Exception):
    pass


class UnknownStateError(Exception):
    pass
