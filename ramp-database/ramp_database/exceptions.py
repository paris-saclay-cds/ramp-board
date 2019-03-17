"""
The :mod:`ramp_database.exceptions` module include all custom warnings and
errors used in ``ramp-database``.
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
    """Error to raise when a submission is already present in the database."""
    pass


class MergeTeamError(Exception):
    """Error to raise when the merging of teams is failing."""
    pass


class MissingSubmissionFileError(Exception):
    """Error to raise when the file submitted is not present in the supposed
    location."""
    pass


class MissingExtensionError(Exception):
    """Error to raise when the extension is not registered in the database."""
    pass


class NameClashError(Exception):
    """Error to raise when there is a duplicate in submission name."""
    pass


class TooEarlySubmissionError(Exception):
    """Error to raise when a submission was submitted to early."""
    pass


class UnknownStateError(Exception):
    """Error to raise when the state of the submission is unknown."""
    pass
