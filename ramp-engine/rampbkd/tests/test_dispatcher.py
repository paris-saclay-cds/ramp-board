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

from rampbkd.local import CondaEnvWorker

from rampbkd.dispatcher import Dispatcher


@pytest.fixture
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
    caplog.set_level(logging.INFO)
    local_tmp_dir = '/tmp'
    # with tempfile.TemporaryDirectory() as local_tmp_dir:
    dispatcher_config = {
        'event_name': 'iris_test',
        'conda_env': 'ramp-iris',
        'local_log_folder': os.path.join(local_tmp_dir, 'log'),
        'local_predictions_folder': os.path.join(local_tmp_dir, 'preds')
    }
    dispatcher = Dispatcher(config=dispatcher_config,
                            worker=CondaEnvWorker, n_worker=-1,
                            worker_policy='exit')
    dispatcher.launch()

    # the iris kit contain a submission which should fail for each user
    submissions = get_submissions(
        event_name=dispatcher_config['event_name'],
    )
    is_submission_failed = ['error' in sub.state for sub in submissions]
    assert sum(is_submission_failed) == 2
