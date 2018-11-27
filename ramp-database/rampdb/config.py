import yaml

DB_SECTION = 'sqlalchemy'
DB_MANDATORY_KEYS = [
    'drivername',
    'username',
    'password',
    'host',
    'port',
    'database',
]
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



def read_database_config(config_file):
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
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    if DB_SECTION not in config:
        raise ValueError(
            "Missing '{}' section in config".format(DB_SECTION))

    section_config = config[DB_SECTION]

    missing_keys = [
        '.'.join(DB_SECTION, key)
        for key in DB_MANDATORY_KEYS
        if key not in section_config
    ]

    if missing_keys:
        missing_keys_str = ', '.join(missing_keys)
        raise ValueError(
            "Missing '{}' key(s) in config".format(missing_keys_str))

    return section_config
