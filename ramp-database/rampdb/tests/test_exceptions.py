import pytest

from rampdb.exceptions import UnknownStateError


def test_unknown_state_error():
    with pytest.raises(UnknownStateError):
        raise UnknownStateError
