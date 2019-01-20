import os
import shutil

from rampdb.testing import setup_files_extension_type
from rampdb.testing import setup_ramp_kits_ramp_data
from rampdb.tools.event import add_event
from rampdb.tools.event import add_problem
from rampdb.tools.event import get_problem
from rampdb.utils import session_scope

from .config_parser import read_config
from .ramp import generate_ramp_config


def deploy_ramp_event(config, setup_ramp_repo=True, force=False):
    """Deploy a RAMP event using a configuration file.

    This utility is in charge of creating the kit and data repository for a
    given RAMP event. It will also setup the database.

    Parameters
    ----------
    config : str
        The path to the configuration file containing all information necessary
        to deploy a RAMP event.
    setup_ramp_repo : bool, default is True
        Whether or not to setup the RAMP kit and data repositories.
    force : bool, default is False
        Whether or not to potentially overwrite the repositories, problem and
        event in the database.
    """
    config = read_config(config)
    database_config = config['sqlalchemy']
    ramp_config = generate_ramp_config(config)

    with session_scope(database_config) as session:
        setup_files_extension_type(session)
        if setup_ramp_repo:
            setup_ramp_kits_ramp_data(config, ramp_config['event'], force)
        # check if the repository exists
        problem = get_problem(session, ramp_config['event'])
        if problem is None:
            add_problem(session, ramp_config['event'],
                        ramp_config['ramp_kits_dir'],
                        ramp_config['ramp_data_dir'])
        else:
            if ((ramp_config['ramp_kits_dir'] != problem.path_ramp_kits or
                 ramp_config['ramp_data_dir'] != problem.path_ramp_data) and
                    not force):
                raise ValueError(
                    'The RAMP problem already exists in the database. The path'
                    'to the kit or to the data is different. You need to set'
                    '"force=True" if you want to overwrite these parameters.'
                )
            if setup_ramp_repo:
                setup_ramp_kits_ramp_data(config, ramp_config['event'], force)
            add_problem(session, ramp_config['event'],
                        ramp_config['ramp_kits_dir'],
                        ramp_config['ramp_data_dir'],
                        force)

        if not os.path.exists(ramp_config['ramp_submissions_dir']):
            os.makedirs(ramp_config['ramp_submissions_dir'])
        add_event(session, ramp_config['event'],
                  ramp_config['event_name'],
                  ramp_config['event_title'],
                  ramp_config['sandbox_name'],
                  ramp_config['ramp_submissions_dir'],
                  ramp_config['event_is_public'],
                  force)
