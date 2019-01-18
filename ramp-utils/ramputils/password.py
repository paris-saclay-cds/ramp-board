import bcrypt

from .string_encoding import encode_string


def hash_password(password):
    """Hash a password.

    Parameters
    ----------
    password : str or bytes
        Human readable password.

    Returns
    -------
    hashed_password : bytes
        The hashed password.
    """
    return bcrypt.hashpw(encode_string(password), bcrypt.gensalt())


def check_password(password, hashed_password):
    """Check if a password is the same than the hashed password.

    Parameters
    ----------
    password : str or bytes
        Human readable password.
    hashed_password : str or bytes
        The hashed password.

    Returns
    -------
    is_same_password : bool
        Return True if the two passwords are identical.
    """
    return bcrypt.checkpw(encode_string(password),
                          encode_string(hashed_password))
