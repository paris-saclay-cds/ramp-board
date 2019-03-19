import six

from .config_parser import read_config
from .ramp import generate_ramp_config


def generate_worker_config(event_config, database_config=None):
    """Generate the configuration for RAMP worker from a configuration
    file.

    Parameters
    ----------
    event_config : dict or str
        Either the loaded configuration or the configuration YAML file. When
        the configuration filename is given, ``database_config`` need to be
        given as well. When a ``dict`` is provided, all paths should be given.
    database_config : str, optional
        The database configuration filename. It is required when
        ``event_config`` is a ``str``..

    Returns
    -------
    worker_config : dict
        The configuration for the RAMP worker.
    """
    if isinstance(event_config, six.string_types):
        ramp_config = generate_ramp_config(event_config, database_config)
        event_config = read_config(
            event_config, filter_section=['ramp', 'worker'])
    else:
        ramp_config = generate_ramp_config(event_config)

    # copy the specific information for the given worker configuration
    worker_config = event_config['worker'].copy()
    # define the directory of the ramp-kit for the event
    worker_config['kit_dir'] = ramp_config['ramp_kit_dir']
    # define the directory of the ramp-data for the event
    worker_config['data_dir'] = ramp_config['ramp_data_dir']
    # define the directory of the submissions
    worker_config['submissions_dir'] = ramp_config['ramp_submissions_dir']
    # define the directory of the predictions
    worker_config['predictions_dir'] = ramp_config['ramp_predictions_dir']
    # define the directory of the logs
    worker_config['logs_dir'] = ramp_config['ramp_logs_dir']

    return worker_config
