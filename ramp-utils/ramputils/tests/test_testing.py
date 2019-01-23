from os.path import join

import pytest

from ramputils import read_config

from ramputils.testing import database_config_template
from ramputils.testing import flask_config_template
from ramputils.testing import path_config_example
from ramputils.testing import ramp_config_template


@pytest.mark.parametrize(
    "config_func, partial_path",
    [(path_config_example, join('tests', 'data', 'config.yml')),
     (database_config_template, join('template', 'database_config.yml')),
     (flask_config_template, join('template', 'flask_config.yml')),
     (ramp_config_template, join('template', 'ramp_config.yml'))]
)
def test_path_configuration(config_func, partial_path):
    path = config_func()
    assert partial_path in path
    read_config(path)
