# TODO: journaling wit git?


from enum import Enum
from databoard import app
from contextlib import contextmanager
from flask.ext.shelve import init_app
from flask.ext.shelve import get_shelve

app.config['SHELVE_FILENAME'] = 'shelve'
app.config['SHELVE_FLAG'] = 'c'
app.config['SHELVE_PROTOCOL'] = 0
app.config['SHELVE_WRITEBACK'] = True

init_app(app)

columns = ['team', 'model', 'timestamp', 'path', 'state', 'listing']


@contextmanager
def shelve_database(flag='c'):
    with app.test_request_context():
        yield get_shelve(flag)

ModelState = Enum('ModelState', 'new trained error')