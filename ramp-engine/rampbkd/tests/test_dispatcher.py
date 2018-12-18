import os
import shutil
import tempfile

import pytest

from databoard import db
from databoard import deployment_path
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


def test_dispatcher(setup_db):
    with tempfile.TemporaryDirectory() as local_tmp_dir:
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
