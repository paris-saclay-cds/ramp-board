from itertools import takewhile
import importlib
import os
import sys

import bcrypt


def import_module_from_source(source, name):
    """Load a module from a Python source file.

    Parameters
    ----------
    source : str
        Path to the Python source file which will be loaded as a module.
    name : str
        Name to give to the module once loaded.
    Returns
    -------
    module : Python module
        Return the Python module which has been loaded.
    """
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _encode_string(text):
    return bytes(text, 'utf-8') if isinstance(text, str) else text


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
    return bcrypt.hashpw(_encode_string(password), bcrypt.gensalt())


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
    return bcrypt.checkpw(
        _encode_string(password), _encode_string(hashed_password)
    )
