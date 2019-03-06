import os

import pytest

from ramp_utils.testing import ramp_config_template

from ramp_utils import read_config
from ramp_utils import generate_ramp_config


@pytest.mark.parametrize(
    "config",
    [ramp_config_template(),
     read_config(ramp_config_template()),
     read_config(ramp_config_template(), filter_section='ramp')]
)
def test_generate_ramp_config(config):
    ramp_config = generate_ramp_config(config)
    expected_config = {
        'problem_name': 'iris',
        'event_name': 'iris_test',
        'event_title': 'Iris event',
        'event_is_public': True,
        'sandbox_name': 'starting_kit',
        'deployment_dir': '/tmp/databoard_test',
        'ramp_kit_dir': os.path.join(
            '/tmp/databoard_test', 'ramp-kits', 'iris'
        ),
        'ramp_data_dir': os.path.join(
            '/tmp/databoard_test', 'ramp-data', 'iris'
        ),
        'ramp_kit_submissions_dir': os.path.join(
            '/tmp/databoard_test', 'ramp-kits', 'iris', 'submissions'
        ),
        'ramp_submissions_dir': os.path.join(
            '/tmp/databoard_test', 'submissions'
        ),
        'ramp_sandbox_dir': os.path.join(
            '/tmp/databoard_test', 'ramp-kits', 'iris', 'submissions',
            'starting_kit'
        )
    }
    assert ramp_config == expected_config
