"""
RAMP main database configuration

This file is currently inherited from ramp-board
"""
from __future__ import absolute_import, unicode_literals

import os
import sys
import yaml

MANDATORY_KEYS = ['sqlalchemy']
MANDATORY_URL_KEYS = ['drivername', 'username', 'password',
                      'host', 'port', 'database']
STATES = [
    'new',               # submitted by user to frontend server
    'checked',           # not used, checking is part of the workflow now
    'checking_error',    # not used, checking is part of the workflow now
    'trained',           # training finished normally on the backend server
    'training_error',    # training finished abnormally on the backend server
    'validated',         # validation finished normally on the backend server
    'validating_error',  # validation finished abnormally on the backend server
    'tested',            # testing finished normally on the backend server
    'testing_error',     # testing finished abnormally on the backend server
    'training',          # training is running normally on the backend server
    'sent_to_training',  # frontend server sent submission to backend server
    'scored',            # submission scored on the frontend server.Final state
]
"""list of int: enumeration of available submission states"""


class UnknownStateError(Exception):
    pass


def read_backend_config(config_file):
    """Parse YAML configuration file

    Parameters
    ----------
    config_file : str
        path to the ramp-backend YAML configuration file

    Returns
    -------
    config : dict
        dictionary with RAMP database configuration info

    Raises
    ------
    ValueError : when mandatory parameters are missing

    """
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    except (OSError, IOError):
        print('Config file not found')
        sys.exit(1)

    missing_keys = [key
                    for key in MANDATORY_KEYS
                    if key not in config]

    missing_keys.extend(['sqlalchemy.' + key
                         for key in MANDATORY_URL_KEYS
                         if key not in config['sqlalchemy']])

    if missing_keys:
        raise ValueError("Missing '{}' value in config"
                         .format(', '.join(missing_keys)))

    return config


deployment_path = '/tmp/databoard'  # edit this!
ramp_kits_path = os.path.join(deployment_path, 'ramp-kits')
ramp_data_path = os.path.join(deployment_path, 'ramp-data')
submissions_d_name = 'submissions'
submissions_path = os.path.join(deployment_path, submissions_d_name)

sandbox_d_name = 'starting_kit'
starting_kit_d_name = 'starting_kit'
sandbox_path = os.path.join(deployment_path, sandbox_d_name)
problems_d_name = 'problems'
problems_path = os.path.join(deployment_path, problems_d_name)

specific_module = 'databoard.specific'
workflows_module = specific_module + '.workflows'
problems_module = specific_module + '.problems'
events_module = specific_module + '.events'
score_types_module = specific_module + '.score_types'
