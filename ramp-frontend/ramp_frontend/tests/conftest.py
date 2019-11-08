import pytest
from sqlalchemy import create_engine, exc
import contextlib

@pytest.fixture(scope='session')
def database():
    '''
    Create a Postgres database for the tests, and drop it when the tests are done.
    '''
    with contextlib.suppress(exc.ProgrammingError):
        engine = create_engine('postgresql:///postgres', isolation_level='AUTOCOMMIT')
        connection = engine.connect()
        try: 
            connection.execute("CREATE USER mrramp WITH PASSWORD 'mrramp';ALTER USER mrramp WITH SUPERUSER")
        except:
            print('mrramp already exists')
        connection.execute('CREATE DATABASE databoard_test OWNER mrramp')

    @request.addfinalizer
    def drop_database():
        connection.execute('DROP DATABASE data_test')
        connection.close()