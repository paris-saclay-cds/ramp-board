import os

import six

from .config_parser import read_config


def generate_worker_config(config):
    """Generate the configuration for RAMP worker from a configuration
    file.

    Parameters
    ----------
    config : dict or str
        Either the loaded configuration or the configuration YAML file.

    Returns
    -------
    worker_config : dict
        The configuration for the RAMP worker.
    """
    if isinstance(config, six.string_types):
        config = read_config(config, filter_section=['ramp', 'worker'])

    ramp_config = config['ramp']

    # copy the specific information for the given worker configuration
    worker_config = config['worker'].copy()
    # define the directory of the ramp-kit for the event
    worker_config['kit_dir'] = os.path.join(
        ramp_config['deployment_dir'],
        ramp_config['kits_dir'],
        ramp_config['event']
    )
    # define the directory of the ramp-data for the event
    worker_config['data_dir'] = os.path.join(
        ramp_config['deployment_dir'],
        ramp_config['data_dir'],
        ramp_config['event']
    )
    # define the directory of the submissions
    worker_config['submissions_dir'] = os.path.join(
        ramp_config['deployment_dir'],
        ramp_config['submissions_dir']
    )
    # define the directory of the predictions
    worker_config['predictions_dir'] = os.path.join(
        ramp_config['deployment_dir'],
        ramp_config['predictions_dir']
    )
    # define the directory of the logs
    worker_config['logs_dir'] = os.path.join(
        ramp_config['deployment_dir'],
        ramp_config['logs_dir']
    )

    return worker_config
