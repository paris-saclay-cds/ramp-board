import asyncore
import contextlib
import os
import pytest
import smtpd
from sqlalchemy import create_engine, exc
from threading import Thread


@pytest.fixture(scope='session')
def database_connection():
    '''
    Create a Postgres database for the tests,
    and drop it when the tests are done.
    '''
    #os.system('pg_ctl -D postgres_dbs -l logfile start')
    #os.system('pg_ctl -D postgres -U postgres -l logfile start')
    #engine = create_engine('postgresql://mtelencz:@localhost/postgres', 
    #                       isolation_level='AUTOCOMMIT')
    engine = create_engine('postgresql://postgres:@localhost/postgres',
                           isolation_level='AUTOCOMMIT')
    connection = engine.connect()

    try:
        connection.execute("""CREATE USER mrramp WITH PASSWORD 'mrramp';
                              ALTER USER mrramp WITH SUPERUSER""")
    except exc.ProgrammingError:
        print('mrramp already exists. Working with existing mrramp')

    try:
        connection.execute('CREATE DATABASE databoard_test OWNER mrramp')
    except exc.ProgrammingError:
        print('database exists. Reusing existing database')

    # close the connection and remove the database in the end
    yield
    connection.execute("""SELECT pg_terminate_backend(pid)
                        FROM pg_stat_activity
                        WHERE datname = 'databoard_test';""")
    connection.execute('DROP DATABASE databoard_test')
    print("deleted database 'databoard_test'")
    