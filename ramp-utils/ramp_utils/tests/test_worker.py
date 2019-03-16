import os

import pytest

from ramp_utils.testing import ramp_config_template
from ramp_utils.testing import database_config_template

from ramp_utils import generate_worker_config


def test_generate_worker_config():
    worker_config = generate_worker_config(
        ramp_config_template(), database_config_template()
    )
    expected_config = {
        'worker_type': 'conda',
        'conda_env': 'ramp-iris',
        'kit_dir': os.path.join('/tmp/databoard_test', 'ramp-kits', 'iris'),
        'data_dir': os.path.join('/tmp/databoard_test', 'ramp-data', 'iris'),
        'submissions_dir': os.path.join('/tmp/databoard_test', 'submissions'),
        'predictions_dir': os.path.join('/tmp/databoard_test', 'preds'),
        'logs_dir': os.path.join('/tmp/databoard_test', 'log')
    }
    assert worker_config == expected_config
