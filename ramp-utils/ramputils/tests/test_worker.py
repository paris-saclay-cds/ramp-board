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
        'conda_env': 'ramp-iris',
        'sandbox_dir': 'starting_kit',
        'kit_dir': os.path.join('/tmp/ramp', 'kits', 'iris'),
        'data_dir': os.path.join('/tmp/ramp', 'data', 'iris'),
        'submissions_dir': os.path.join('/tmp/ramp', 'submissions'),
        'predictions_dir': os.path.join('/tmp/ramp', 'preds'),
        'logs_dir': os.path.join('/tmp/ramp', 'log')
    }
    assert worker_config == expected_config
