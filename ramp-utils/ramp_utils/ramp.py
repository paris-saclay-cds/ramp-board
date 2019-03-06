import os

import six

from .config_parser import read_config


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
    ramp_config = {}
    ramp_config['event'] = config['event']
    ramp_config['event_name'] = config['event_name']
    ramp_config['event_title'] = config['event_title']
    ramp_config['event_is_public'] = config['event_is_public']
    ramp_config['sandbox_name'] = config['sandbox_dir']
    ramp_config['ramp_kit_dir'] = config['kit_dir']
    ramp_config['ramp_data_dir'] = config['data_dir']
    ramp_config['ramp_kit_submissions_dir'] = os.path.join(
        ramp_config['ramp_kit_dir'], 'submissions'
    )
    ramp_config['ramp_submissions_dir'] = config['submissions_dir']
    ramp_config['ramp_sandbox_dir'] = os.path.join(
        ramp_config['ramp_kit_dir'], 'submissions', config['sandbox_dir']
    )
    return ramp_config
