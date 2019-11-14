import contextlib
import os
import pytest
import smtpd
from sqlalchemy import create_engine, exc
from threading import Thread

from yaml import safe_load


@pytest.fixture(scope='session')
def database_connection():
    '''
    Create a Postgres database for the tests,
    and drop it when the tests are done.
    '''
    #os.system('pg_ctl -D postgres -U postgres -l logfile start')

    #engine = create_engine('postgresql://<local_user>:@localhost/<engine_name>', 
    #                       isolation_level='AUTOCOMMIT')
    config = safe_load(open("db_engine.yml"))
    dbowner = config.get('db_owner')
    dbcluster = config.get('db_cluster_name')
    engine = create_engine(f'postgresql://{dbowner}:@localhost/{dbcluster}', 
                           isolation_level='AUTOCOMMIT')
    #engine = create_engine('postgresql://postgres:@localhost/import argparse',
    #                       isolation_level='AUTOCOMMIT')
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
    connection.execute('DROP USER mrramp')
    print("deleted database 'databoard_test' and removed user 'mrramp'")
    