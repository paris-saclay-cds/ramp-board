from ramputils.testing import path_config_example
from ramputils import read_config


def test_read_config():
    config_file = path_config_example()
    config = read_config(config_file)
    expected_config = {
        'sqlalchemy': {
            'drivername': 'postgresql',
            'username': 'mrramp',
            'password': 'mrramp',
            'host': 'localhost',
            'port': 5432,
            'database': 'databoard_test'
        }
    }
    assert config == expected_config
