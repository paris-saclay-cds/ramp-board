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
