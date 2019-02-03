import pytest

from ramp_database.exceptions import DuplicateSubmissionError
from ramp_database.exceptions import MergeTeamError
from ramp_database.exceptions import MissingExtensionError
from ramp_database.exceptions import MissingSubmissionFileError
from ramp_database.exceptions import NameClashError
from ramp_database.exceptions import TooEarlySubmissionError
from ramp_database.exceptions import UnknownStateError


@pytest.mark.parametrize(
    "ExceptionClass",
    [DuplicateSubmissionError,
     MergeTeamError,
     MissingExtensionError,
     MissingSubmissionFileError,
     NameClashError,
     TooEarlySubmissionError,
     UnknownStateError]
)
def test_exceptions(ExceptionClass):
    with pytest.raises(ExceptionClass):
        raise ExceptionClass


@pytest.mark.parametrize(
    "ExceptionClass",
    [DuplicateSubmissionError,
     MergeTeamError,
     MissingExtensionError,
     MissingSubmissionFileError,
     NameClashError,
     TooEarlySubmissionError,
     UnknownStateError]
)
def test_exceptions_msg(ExceptionClass):
    with pytest.raises(ExceptionClass, match='Some error message'):
        raise ExceptionClass('Some error message')


@pytest.mark.parametrize(
    "ExceptionClass",
    [DuplicateSubmissionError,
     MergeTeamError,
     MissingExtensionError,
     MissingSubmissionFileError,
     NameClashError,
     TooEarlySubmissionError,
     UnknownStateError]
)
def test_exceptions_obj(ExceptionClass):
    class DummyObject:
        pass

    with pytest.raises(ExceptionClass):
        raise ExceptionClass(DummyObject())
