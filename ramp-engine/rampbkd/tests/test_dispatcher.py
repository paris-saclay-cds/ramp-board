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

from databoard import db
from databoard import deployment_path
from databoard.db_tools import get_submissions
from databoard.testing import create_toy_db

from ramputils import read_config
from ramputils.testing import path_config_example

from rampbkd.local import CondaEnvWorker
from rampbkd.dispatcher import Dispatcher


@pytest.fixture(scope='module')
def setup_db():
    try:
        create_toy_db()
        yield
    finally:
        shutil.rmtree(deployment_path, ignore_errors=True)
        db.session.close()
        db.session.remove()
        db.drop_all()


def test_dispatcher(setup_db, caplog):
    config = read_config(path_config_example())
    # with tempfile.TemporaryDirectory() as local_tmp_dir:
    dispatcher = Dispatcher(config=config,
                            worker=CondaEnvWorker, n_worker=-1,
                            hunger_policy='exit')
    dispatcher.launch()

    # the iris kit contain a submission which should fail for each user
    submissions = get_submissions(
        event_name=config['ramp']['event_name'],
    )
    is_submission_failed = ['error' in sub.state for sub in submissions]
    assert sum(is_submission_failed) == 2
