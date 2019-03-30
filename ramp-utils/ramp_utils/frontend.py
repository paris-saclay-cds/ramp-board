from .config_parser import read_config


DEFAULT_CONFIG = {
    'WTF_CSRF_ENABLED': True,
    'LOG_FILENAME': 'None',
    'MAX_CONTENT_LENGTH': 1073741824,
    'SQLALCHEMY_TRACK_MODIFICATIONS': False,
    'TRACK_USER_INTERACTION': False,
    'DOMAIN_NAME': 'localhost'
}


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
        config = read_config(config, filter_section=['flask', 'sqlalchemy'])

    flask_config = DEFAULT_CONFIG.copy()
    user_flask_config = {
        key.upper(): value for key, value in config['flask'].items()}
    flask_config.update(user_flask_config)

    database_config = config['sqlalchemy']
    flask_config['SQLALCHEMY_DATABASE_URI'] = \
        ('{}://{}:{}@{}:{}/{}'
         .format(database_config['drivername'], database_config['username'],
                 database_config['password'], database_config['host'],
                 database_config['port'], database_config['database']))
    return flask_config
