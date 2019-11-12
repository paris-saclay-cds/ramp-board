import pytest
from sqlalchemy import create_engine, exc
import contextlib
import smtpd
import asyncore
from threading import Thread

@pytest.fixture(scope='session')
def connection():
    '''
    Create a Postgres database for the tests, and drop it when the tests are done.
    '''

    engine = create_engine('postgresql:///postgres', isolation_level='AUTOCOMMIT')
    connection = engine.connect()
    try: 
        connection.execute("CREATE USER mrramp WITH PASSWORD 'mrramp';ALTER USER mrramp WITH SUPERUSER")
    except exc.ProgrammingError:
        print('mrramp already exists')
    try:
        connection.execute('CREATE DATABASE databoard_test OWNER mrramp')
    except exc.ProgrammingError:
        print('database exists. Reusing existing database')
       
    yield
    connection.execute("""SELECT pg_terminate_backend(pid)
    FROM pg_stat_activity
    WHERE datname = 'databoard_test';""")
    connection.execute('DROP DATABASE databoard_test')
    print("deleted database 'databoard_test'")


class SMTPServerThread(Thread):
    def __init__(self):
        super().__init__()

    def run(self):

        self.smtp = smtpd.DebuggingServer(('localhost', 8025),None)

        asyncore.loop(timeout=0.1)

    def close(self):
        self.smtp.close()
        
@pytest.fixture(scope='session')
def smtp_server():
    '''
    Creates the smtp server
    '''
   

    server = SMTPServerThread()
    server.run()
    
    yield server

    server.close()
    server.join(1)