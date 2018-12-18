import os
import shutil

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
    dispatcher_config = {
        'event_name': 'iris_test',
        'conda_env': 'ramp-iris',
        'local_log_folder': os.path.join('/tmp', 'log'),
        'local_predictions_folder': os.path.join('/tmp', 'preds')
    }
    dispatcher = Dispatcher(config=dispatcher_config, worker=CondaEnvWorker,
                            n_worker=-1, worker_policy='exit')
    # dispatcher.launch()
    try:
        dispatcher.launch()
    finally:
        for path in ['log', 'predictions']:
            shutil.rmtree(dispatcher_config['local_{}_folder'.format(path)],
                          ignore_errors=True)