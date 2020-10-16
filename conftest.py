import contextlib
import os
import pytest
import smtpd
from sqlalchemy import create_engine, exc
from threading import Thread
from ramp_utils.testing import database_config_template
from yaml import safe_load

from ramp_utils import read_config


@pytest.fixture(scope='session')
def database_connection():
    """
    Create a Postgres database for the tests, and drop it when the tests are
    done.
    """
    config = safe_load(open("db_engine.yml"))
    dbowner = config.get('db_owner')

    engine = create_engine(f'postgresql://{dbowner}:@localhost/postgres',
                           isolation_level='AUTOCOMMIT')

    connection = engine.connect()

    database_config = read_config(database_config_template())
    username = database_config['sqlalchemy']['username']
    database_name = database_config['sqlalchemy']['database']
    try:
        connection.execute(f"""CREATE USER {username}
                              WITH PASSWORD '{username}';
                              ALTER USER {username} WITH SUPERUSER""")
    except exc.ProgrammingError as e:
        raise ValueError(f'user {username} already exists') from e

    try:
        connection.execute(f'CREATE DATABASE {database_name} OWNER {username}')
    except exc.ProgrammingError as e:
        raise ValueError(
            f'{database_name} database used for testing already exists'
        ) from e

    # close the connection and remove the database in the end
    yield
    connection.execute("""SELECT pg_terminate_backend(pid)
                       FROM pg_stat_activity
                       WHERE datname = 'databoard_test';""")
    connection.execute(f'DROP DATABASE {database_name}')
    connection.execute(f'DROP USER {username}')
    print(f"deleted database 'databoard_test' and removed user '{username}'")
