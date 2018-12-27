import os

import pytest

from ramputils.testing import path_config_example

from ramputils import read_config
from ramputils import generate_ramp_config


@pytest.mark.parametrize(
    "config",
    [path_config_example(),
     read_config(path_config_example()),
     read_config(path_config_example(), filter_section='ramp')]
)
def test_generate_ramp_config(config):
    ramp_config = generate_ramp_config(config)
    expected_config = {
        'event': 'iris',
        'event_name': 'iris_test',
        'deployment_dir': '/tmp/databoard_test',
        'ramp_kits_dir': os.path.join('/tmp/databoard_test', 'ramp-kits'),
        'ramp_data_dir': os.path.join('/tmp/databoard_test', 'ramp-data'),
        'ramp_submissions_dir': os.path.join('/tmp/databoard_test',
                                             'submissions'),
        'ramp_sandbox_dir': os.path.join('/tmp/databoard_test', 'ramp-kits',
                                         'iris', 'submissions', 'starting_kit')
    }
    assert ramp_config == expected_config
