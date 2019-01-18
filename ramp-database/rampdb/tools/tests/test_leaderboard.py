import datetime
import shutil

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampbkd.dispatcher import Dispatcher

from rampdb.model import Model

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.leaderboard import get_leaderboard


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture(scope='module')
def session_toy_db(config):
    try:
        create_toy_db(config)
        dispatcher = Dispatcher(config=config, n_worker=-1,
                                hunger_policy='exit')
        dispatcher.launch()
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, _ = setup_db(config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_get_leaderboard(session_toy_db):
    print(get_leaderboard(session_toy_db, 'public', 'iris_test'))
#     print(get_leaderboard(session_toy_db, 'private', 'iris_test'))
#     print(get_leaderboard(session_toy_db, 'new', 'iris_test'))
#     print(get_leaderboard(session_toy_db, 'failed', 'iris_test'))
