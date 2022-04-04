from .config_parser import read_config


DEFAULT_CONFIG = {
    "WTF_CSRF_ENABLED": True,
    "LOG_FILENAME": "None",
    "MAX_CONTENT_LENGTH": 1073741824,
    "SQLALCHEMY_TRACK_MODIFICATIONS": False,
    "TRACK_USER_INTERACTION": False,
    "TRACK_CREDITS": False,
    "DOMAIN_NAME": "localhost",
    "LOGIN_INSTRUCTIONS": None,
    "SIGN_UP_INSTRUCTIONS": None,
    "SIGN_UP_ASK_SOCIAL_MEDIA": False,
    "PRIVACY_POLICY_PAGE": None,
    "THREADPOOL_MAX_WORKERS": 2,
}


def _read_if_html_path(txt: str) -> str:
    """Open HTML file if path provided

    If the input is a path to a valid HTML file, read it.
    Otherwise return the input
    """
    if txt and txt.endswith(".html"):
        with open(txt, "rt") as fh:
            txt = fh.read()
    return txt


def generate_flask_config(config):
    """Generate the configuration to deal with Flask.

    Parameters
    ----------
    config : dict or str
        Either the loaded configuration or the configuration YAML file.

    Returns
    -------
    flask_config : dict
        The configuration for the RAMP worker.
    """
    if isinstance(config, str):
        config = read_config(config, filter_section=["flask", "sqlalchemy"])

    flask_config = DEFAULT_CONFIG.copy()
    user_flask_config = {key.upper(): value for key, value in config["flask"].items()}
    flask_config.update(user_flask_config)

    for key in [
        "LOGIN_INSTRUCTIONS",
        "SIGN_UP_INSTRUCTIONS",
        "PRIVACY_POLICY_PAGE",
    ]:
        flask_config[key] = _read_if_html_path(flask_config[key])

    database_config = config["sqlalchemy"]
    flask_config["SQLALCHEMY_DATABASE_URI"] = "{}://{}:{}@{}:{}/{}".format(
        database_config["drivername"],
        database_config["username"],
        database_config["password"],
        database_config["host"],
        database_config["port"],
        database_config["database"],
    )
    return flask_config
