import os

import six

from .config_parser import read_config


def _create_default_path(config, key, path_config):
    default_mapping = {
        'kit_dir': os.path.join(
            path_config, 'ramp-kits', config['problem_name']
        ),
        'data_dir': os.path.join(
            path_config, 'ramp-data', config['problem_name']
        ),
        'submissions_dir': os.path.join(
            path_config, 'submissions'
        ),
        'sandbox_dir': 'starting_kit',
        'predictions_dir': os.path.join(
            path_config, 'predictions'
        ),
        'logs_dir': os.path.join(
            path_config, 'logs'
        )
    }
    if key not in config:
        return default_mapping[key]
    return config[key]


def generate_ramp_config(config):
    """Generate the configuration to deploy RAMP.

    Parameters
    ----------
    config : dict or str
        Either the loaded configuration or the configuration YAML file.

    Returns
    -------
    ramp_config : dict
        The configuration for the RAMP worker.
    """
    if isinstance(config, six.string_types):
        config = read_config(config, filter_section='ramp')
    else:
        if 'ramp' in config.keys():
            config = config['ramp']
    path_config = os.getcwd()

    ramp_config = {}
    # mandatory parameters
    ramp_config['problem_name'] = config['problem_name']
    ramp_config['event_name'] = config['event_name']
    ramp_config['event_title'] = config['event_title']
    ramp_config['event_is_public'] = config['event_is_public']

    # parameter which can built by default
    ramp_config['ramp_kit_dir'] = _create_default_path(
        config, 'kit_dir', path_config
    )
    ramp_config['ramp_data_dir'] = _create_default_path(
        config, 'data_dir', path_config
    )
    ramp_config['ramp_submissions_dir'] = _create_default_path(
        config, 'submissions_dir', path_config
    )
    ramp_config['sandbox_name'] = _create_default_path(
        config, 'sandbox_dir', ''
    )
    ramp_config['ramp_predictions_dir'] = _create_default_path(
        config, 'predictions_dir', path_config
    )
    ramp_config['ramp_logs_dir'] = _create_default_path(
        config, 'logs_dir', path_config
    )

    # parameters built on the top of the previous one
    ramp_config['ramp_sandbox_dir'] = os.path.join(
        ramp_config['ramp_kit_dir'], 'submissions', ramp_config['sandbox_name']
    )
    ramp_config['ramp_kit_submissions_dir'] = os.path.join(
        ramp_config['ramp_kit_dir'], 'submissions'
    )
    return ramp_config
