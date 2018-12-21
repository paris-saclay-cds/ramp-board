import os

from ramputils.testing import path_config_example


def test_path_config_example():
    path = path_config_example()
    assert os.path.join('tests', 'data', 'config.yml') in path
