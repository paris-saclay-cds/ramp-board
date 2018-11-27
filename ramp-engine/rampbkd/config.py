"""
RAMP main database configuration

This file is currently inherited from ramp-board
"""
from __future__ import absolute_import, unicode_literals

import os
import sys
import yaml

SANDBOX_NAME = 'starting_kit'
MANDATORY_KEYS = ['sqlalchemy']
MANDATORY_URL_KEYS = ['drivername', 'username', 'password',
                      'host', 'port', 'database']


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

    if 'test' in config:
        os.environ['RAMP_SERVER_TYPE'] = 'TEST'
    else:
        os.environ['RAMP_SERVER_TYPE'] = 'PROD'

    return config


def get_deployment_path():
    server_type = os.environ.get('RAMP_SERVER_TYPE', 'UNKNOWN')
    if server_type == 'TEST':
        return '/mnt/ramp/frontend'
    elif server_type == 'PROD':
        return '/mnt/ramp_data/frontend'
    else:
        raise AttributeError("The RAMP_SERVER_TYPE environment variable was "
                             "not found.")
