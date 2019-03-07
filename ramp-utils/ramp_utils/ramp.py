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
        'submission_dir': os.path.join(
            path_config, 'submissions'
        ),
        'sandbox_name': 'starting_kit'
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
    path_config = os.path.dirname(os.path.abspath(config))
    config = read_config(config, filter_section='ramp')

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

    # parameters built on the top of the previous one
    ramp_config['ramp_sandbox_dir'] = os.path.join(
        ramp_config['ramp_kit_dir'], 'submissions', ramp_config['sandbox_name']
    )
    ramp_config['ramp_kit_submissions_dir'] = os.path.join(
        ramp_config['ramp_kit_dir'], 'submissions'
    )
    return ramp_config
