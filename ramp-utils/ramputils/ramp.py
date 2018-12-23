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
    ramp_config['deployment_dir'] = config['deployment_dir']
    ramp_config['ramp_kits_dir'] = os.path.join(
        config['deployment_dir'], config['kits_dir']
    )
    ramp_config['ramp_data_dir'] = os.path.join(
        config['deployment_dir'], config['data_dir']
    )
    ramp_config['ramp_submissions_dir'] = os.path.join(
        config['deployment_dir'], config['submissions_dir']
    )
    return ramp_config
