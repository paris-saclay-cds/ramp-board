# TODO: remove this file once we are moving to Python 3 only
import sys

PYTHON3 = sys.version_info[0] == 3


def encode_string(text):
    """Encode text into an array of bytes in both Python 2 and 3 with UTF-8.

    Parameters
    ----------
    text : str or bytes
        The text to be encoded

    Returns
    -------
    encoded_text : bytes
        The encoded text.
    """
    if PYTHON3:
        return bytes(text, 'utf-8') if isinstance(text, str) else text
    return text.encode('utf8')