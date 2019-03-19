import os

import pytest

from ramp_utils.testing import ramp_config_template
from ramp_utils.testing import database_config_template

from ramp_utils import read_config
from ramp_utils import generate_ramp_config

HERE = os.path.dirname(__file__)


def _get_event_config(version):
    return os.path.join(
        HERE, 'data', 'ramp_config_{}.yml'.format(version)
    )


@pytest.mark.parametrize(
    "event_config, database_config, err_msg",
    [(_get_event_config('absolute'), None,
      "you need to provide the filename of the database as well"),
     (read_config(_get_event_config('missing')), None,
      "you need to provide all following keys")]
)
def test_generate_ramp_config_error(event_config, database_config, err_msg):
    with pytest.raises(ValueError, match=err_msg):
        generate_ramp_config(event_config, database_config)


@pytest.mark.parametrize(
    "event_config, database_config",
    [(ramp_config_template(), database_config_template()),
     (read_config(ramp_config_template()), None),
     (read_config(ramp_config_template(), filter_section='ramp'), None)]
)
def test_generate_ramp_config(event_config, database_config):
    ramp_config = generate_ramp_config(event_config, database_config)
    expected_config = {
        'problem_name': 'iris',
        'event_name': 'iris_test',
        'event_title': 'Iris event',
        'event_is_public': True,
        'sandbox_name': 'starting_kit',
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
        ),
        'ramp_logs_dir': os.path.join(
            '/tmp/databoard_test', 'log'
        ),
        'ramp_predictions_dir': os.path.join(
            '/tmp/databoard_test', 'preds'
        )
    }
    assert ramp_config == expected_config


def test_generate_ramp_config_short():
    ramp_config = generate_ramp_config(
        _get_event_config('short'), database_config_template()
    )
    expected_config = {
        'problem_name': 'iris',
        'event_name': 'iris_test',
        'event_title': 'Iris event',
        'ramp_kit_dir': os.path.join('template', 'ramp-kits', 'iris'),
        'ramp_data_dir': os.path.join('template', 'ramp-data', 'iris'),
        'ramp_submissions_dir': os.path.join(
            'template', 'events', 'iris_test', 'submissions'
        ),
        'sandbox_name': 'starting_kit',
        'ramp_predictions_dir': os.path.join(
            'template', 'events', 'iris_test', 'predictions'
        ),
        'ramp_logs_dir': os.path.join(
            'template', 'events', 'iris_test', 'logs'
        ),
        'ramp_sandbox_dir': os.path.join(
            'template', 'ramp-kits', 'iris', 'submissions', 'starting_kit'
        ),
        'ramp_kit_submissions_dir': os.path.join(
            'template', 'ramp-kits', 'iris', 'submissions'
        )
    }
    for key in expected_config:
        assert expected_config[key] in ramp_config[key]
