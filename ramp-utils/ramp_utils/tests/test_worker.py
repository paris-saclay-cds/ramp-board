import os

import pytest

from ramp_utils.testing import ramp_config_template

from ramp_utils import read_config
from ramp_utils import generate_worker_config


@pytest.mark.parametrize(
    "config", [ramp_config_template(), read_config(ramp_config_template())]
)
def test_generate_worker_config(config):
    worker_config = generate_worker_config(config)
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
