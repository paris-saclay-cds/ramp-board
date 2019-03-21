"""
The :mod:`ramp_database.utils` module provides tools to setup and connect to
the RAMP database.
"""

from contextlib import contextmanager

import bcrypt

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine.url import URL

from .model import Model


def setup_db(config):
    """Create a sqlalchemy engine and session to interact with the database.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.

    Returns
    -------
    db : :class:`sqlalchemy.Engine`
        The engine to connect to the database.
    Session : :class:`sqlalchemy.orm.Session`
        Configured Session class which can later be used to communicate with
        the database.
    """
    # create the URL from the configuration
    db_url = URL(**config)
    db = create_engine(db_url)
    Session = sessionmaker(db)
    # Link the relational model to the database
    Model.metadata.create_all(db)

    return db, Session


@contextmanager
def session_scope(config):
    """Connect to a database and provide a session to make some operation.

    Parameters
    ----------
    config : dict
        Configuration file containing the information to connect to the
        dataset. If you are using the configuration provided by ramp, it
        corresponds to the the `sqlalchemy` key.

    Returns
    -------
    session : :class:`sqlalchemy.orm.Session`
        The session to directly perform the operation on the database.
    """
    db, Session = setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        try:
            yield session
            session.commit()
        except:  # noqa
            session.rollback()
            raise
        finally:
            session.close()


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
