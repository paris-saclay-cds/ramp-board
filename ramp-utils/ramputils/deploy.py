import os

from rampdb.testing import setup_files_extension_type
from rampdb.testing import setup_ramp_kits_ramp_data
from rampdb.tools.event import add_event
from rampdb.tools.event import add_problem
from rampdb.tools.event import get_problem
from rampdb.utils import session_scope

from .config_parser import read_config
from .ramp import generate_ramp_config


def deploy_ramp_event(config):
    """Deploy a RAMP event using a configuration file.

    This utility will:

    * create and setup the database;
    * clone the the kit and data.

    Parameters
    ----------
    config : str
        The path to the configuration file containing all information necessary
        to deploy a RAMP event.
    """
    config = read_config(config)
    database_config = config['sqlalchemy']
    ramp_config = generate_ramp_config(config)

    def _create_dir(directory):
        if not os.path.exists(directory):
            os.makedirs(directory)

    _create_dir(ramp_config['ramp_kits_dir'])
    _create_dir(ramp_config['ramp_data_dir'])
    _create_dir(ramp_config['ramp_submissions_dir'])

    with session_scope(database_config) as session:
        setup_files_extension_type(session)
        # check if the problem was already created previously
        if get_problem(session, ramp_config['event']) is None:
            setup_ramp_kits_ramp_data(config, ramp_config['event'])
            add_problem(session, ramp_config['event'],
                        ramp_config['ramp_kits_dir'],
                        ramp_config['ramp_data_dir'])
        add_event(session, ramp_config['event'],
                  ramp_config['event_name'],
                  ramp_config['event_title'],
                  ramp_config['sandbox_name'],
                  ramp_config['ramp_submissions_dir'],
                  ramp_config['event_is_public'],
                  False)
