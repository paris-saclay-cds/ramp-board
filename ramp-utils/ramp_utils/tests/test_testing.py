from os.path import join

import pytest

from ramp_utils import read_config

from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template


@pytest.mark.parametrize(
    "config_func, partial_path",
    [(database_config_template, join('template', 'database_config.yml')),
     (ramp_config_template, join('template', 'ramp_config.yml'))]
)
def test_path_configuration(config_func, partial_path):
    path = config_func()
    assert partial_path in path
    read_config(path)
