from ramputils.testing import path_config_example
from ramputils import read_config


def test_read_config():
    config_file = path_config_example()
    config = read_config(config_file)
    print(config)
