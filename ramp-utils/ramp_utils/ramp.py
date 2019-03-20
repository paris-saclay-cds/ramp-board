import os

from .config_parser import read_config

MANDATORY_DICT_PARAMS = ('kit_dir', 'data_dir', 'submissions_dir',
                         'sandbox_dir', 'predictions_dir', 'logs_dir')


def _create_defaults(config, key, path_config):
    """Create the default values if the key is not in the configuration
    already.
    """
    default_mapping = {
        'kit_dir': os.path.join(
            path_config, 'ramp-kits', config['problem_name']
        ),
        'data_dir': os.path.join(
            path_config, 'ramp-data', config['problem_name']
        ),
        'submissions_dir': os.path.join(
            path_config, 'events', config['event_name'], 'submissions'
        ),
        'predictions_dir': os.path.join(
            path_config, 'events', config['event_name'], 'predictions'
        ),
        'logs_dir': os.path.join(
            path_config, 'events', config['event_name'], 'logs'
        ),
        'sandbox_dir': 'starting_kit'
    }
    if key not in config:
        return default_mapping[key]
    return config[key]


def generate_ramp_config(event_config, database_config=None):
    """Generate the configuration to deploy RAMP.

    Parameters
    ----------
    event_config : dict or str
        Either the loaded configuration or the configuration YAML file. When
        the configuration filename is given, ``database_config`` need to be
        given as well. When a ``dict`` is provided, all paths should be given.
    database_config : str, optional
        The database configuration filename. It is required when
        ``event_config`` is a ``str``.

    Returns
    -------
    ramp_config : dict
        The configuration for the RAMP worker.
    """
    if isinstance(event_config, str):
        if (database_config is None or
                not isinstance(database_config, str)):
            raise ValueError(
                'When "event_config" corresponds to the filename of the '
                'configuration, you need to provide the filename of the '
                'database as well, by assigning "database_config".'
            )
        config = read_config(event_config, filter_section='ramp')
        path_config = os.path.dirname(
            os.path.abspath(database_config)
        )
    else:
        if 'ramp' in event_config.keys():
            config = event_config['ramp']
        else:
            config = event_config
        if not all([key in config.keys() for key in MANDATORY_DICT_PARAMS]):
            raise ValueError(
                'When "event_config" is a dictionary, you need to provide all '
                'following keys: {}'.format(MANDATORY_DICT_PARAMS)
            )
        path_config = ''

    ramp_config = {}
    # mandatory parameters
    ramp_config['problem_name'] = config['problem_name']
    ramp_config['event_name'] = config['event_name']
    ramp_config['event_title'] = config['event_title']
    ramp_config['event_is_public'] = config['event_is_public']

    # parameters which can be built by default if given a string
    ramp_config['ramp_kit_dir'] = _create_defaults(
        config, 'kit_dir', path_config
    )
    ramp_config['ramp_data_dir'] = _create_defaults(
        config, 'data_dir', path_config
    )
    ramp_config['ramp_submissions_dir'] = _create_defaults(
        config, 'submissions_dir', path_config
    )
    ramp_config['sandbox_name'] = _create_defaults(
        config, 'sandbox_dir', ''
    )
    ramp_config['ramp_predictions_dir'] = _create_defaults(
        config, 'predictions_dir', path_config
    )
    ramp_config['ramp_logs_dir'] = _create_defaults(
        config, 'logs_dir', path_config
    )

    # parameters inferred from the previous one
    ramp_config['ramp_sandbox_dir'] = os.path.join(
        ramp_config['ramp_kit_dir'], 'submissions', ramp_config['sandbox_name']
    )
    ramp_config['ramp_kit_submissions_dir'] = os.path.join(
        ramp_config['ramp_kit_dir'], 'submissions'
    )
    return ramp_config
