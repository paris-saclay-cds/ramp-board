import pytest

from rampdb.exceptions import DuplicateSubmissionError
from rampdb.exceptions import MergeTeamError
from rampdb.exceptions import MissingExtensionError
from rampdb.exceptions import MissingSubmissionFileError
from rampdb.exceptions import NameClashError
from rampdb.exceptions import TooEarlySubmissionError
from rampdb.exceptions import UnknownStateError


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
