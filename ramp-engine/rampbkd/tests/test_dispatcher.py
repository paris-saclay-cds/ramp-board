import logging
import os
import shutil
import sys

PYTHON_MAJOR_VERSION = sys.version_info[0]
if PYTHON_MAJOR_VERSION >= 3:
    import tempfile
else:
    from backports import tempfile

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Model
from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db
from rampdb.tools.submission import get_submissions

from rampbkd.local import CondaEnvWorker
from rampbkd.dispatcher import Dispatcher


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture(scope='module')
def session_scope_module(config):
    try:
        create_toy_db(config)
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


def test_dispatcher(session_scope_module, caplog, config):
    # with tempfile.TemporaryDirectory() as local_tmp_dir:
    dispatcher = Dispatcher(config=config,
                            worker=CondaEnvWorker, n_worker=-1,
                            hunger_policy='exit')
    dispatcher.launch()

    # the iris kit contain a submission which should fail for each user
    with session_scope(config['sqlalchemy']) as session:
        submission = get_submissions(session, config['ramp']['event_name'],
                                     'training_error')
        assert len(submission) == 2
