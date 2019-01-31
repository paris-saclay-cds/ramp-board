import os

import pytest

from ramputils.testing import path_config_example

from ramputils import read_config
from ramputils import generate_worker_config


@pytest.mark.parametrize(
    "config", [path_config_example(), read_config(path_config_example())]
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
