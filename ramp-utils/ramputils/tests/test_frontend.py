import pytest

from ramputils.testing import database_config_template

from ramputils import read_config
from ramputils import generate_flask_config


@pytest.mark.parametrize(
    "config",
    [database_config_template(),
     read_config(database_config_template())]
)
def test_generate_flask_config(config):
    flask_config = generate_flask_config(config)
    expected_config = {
        'SECRET_KEY': 'abcdefghijkl',
        'WTF_CSRF_ENABLED': True,
        'LOG_FILENAME': 'None',
        'MAX_CONTENT_LENGTH': 1073741824,
        'DEBUG': True,
        'TESTING': False,
        'MAIL_SERVER': 'localhost',
        'MAIL_PORT': 8025,
        'MAIL_DEFAULT_SENDER': ['RAMP admin', 'rampmailer@localhost.com'],
        'SQLALCHEMY_TRACK_MODIFICATIONS': True,
        'SQLALCHEMY_DATABASE_URI': ('postgresql://mrramp:mrramp@localhost:5432'
                                    '/databoard_test')
        }
    assert flask_config == expected_config
