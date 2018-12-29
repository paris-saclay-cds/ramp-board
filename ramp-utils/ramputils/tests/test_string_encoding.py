import sys

from ramputils import encode_string

PYTHON3 = sys.version_info[0] == 3


def test_encode_string():
    if PYTHON3:
        string = encode_string('a string')
        assert isinstance(string, bytes)
        string = encode_string(b'a string')
        assert isinstance(string, bytes)
    else:
        string = encode_string('a string')
        assert isinstance(string, bytes)
