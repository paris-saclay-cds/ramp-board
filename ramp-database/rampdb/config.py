import yaml

MANDATORY_SECTION = 'sqlalchemy'
MANDATORY_KEYS = [
    'drivername', 'username', 'password', 'host', 'port', 'database']


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

    if MANDATORY_SECTION not in config:
        raise ValueError(
            "Missing '{}' section in config".format(MANDATORY_SECTION))
    else:
        missing_keys = [
            '.'.join(MANDATORY_SECTION, key)
            for key in MANDATORY_KEYS
            if key not in config[MANDATORY_SECTION]
        ]

        if missing_keys:
            missing_keys_str = ', '.join(missing_keys)
            raise ValueError(
                "Missing '{}' key(s) in config".format(missing_keys_str))

    return config