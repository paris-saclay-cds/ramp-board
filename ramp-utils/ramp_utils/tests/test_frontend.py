import pytest

from ramp_utils.testing import database_config_template

from ramp_utils import read_config
from ramp_utils import generate_flask_config
from ramp_utils.frontend import _read_if_html_path


@pytest.mark.parametrize(
    "config",
    [database_config_template(), read_config(database_config_template())],
)
def test_generate_flask_config(config):
    flask_config = generate_flask_config(config)
    expected_config = {
        "SECRET_KEY": "abcdefghijkl",
        "WTF_CSRF_ENABLED": True,
        "LOGIN_INSTRUCTIONS": None,
        "LOG_FILENAME": "None",
        "MAX_CONTENT_LENGTH": 1073741824,
        "PRIVACY_POLICY_PAGE": None,
        "DEBUG": True,
        "TESTING": False,
        "MAIL_SERVER": "localhost",
        "MAIL_PORT": 8025,
        "MAIL_DEFAULT_SENDER": ["RAMP admin", "rampmailer@localhost.com"],
        "SIGN_UP_ASK_SOCIAL_MEDIA": False,
        "SIGN_UP_INSTRUCTIONS": None,
        "SQLALCHEMY_TRACK_MODIFICATIONS": False,
        "SQLALCHEMY_DATABASE_URI": (
            "postgresql://mrramp:mrramp@localhost:5432/databoard_test"
        ),
        "TRACK_USER_INTERACTION": True,
        "TRACK_CREDITS": True,
        "DOMAIN_NAME": "localhost",
        "THREADPOOL_MAX_WORKERS": 2,
    }
    assert flask_config == expected_config


def test_read_if_html_path(tmpdir):
    msg = "some arbitrary text"
    assert _read_if_html_path(msg) == msg

    msg = "an_ivalid_path.html"
    with pytest.raises(FileNotFoundError):
        _read_if_html_path(msg)

    msg = str(tmpdir / "some_file.html")
    with open(msg, "wt") as fh:
        fh.write("Privacy Policy")
    assert _read_if_html_path(msg) == "Privacy Policy"
