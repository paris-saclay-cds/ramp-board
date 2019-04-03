import os
import yaml

REQUIRED_KEYS = {
    'sqlalchemy': {'drivername', 'username', 'password', 'host', 'port',
                   'database'},
    'ramp': {'problem_name', 'event_name', 'event_title', 'event_is_public'},
    'worker': {'worker_type'}
}


def read_config(config_file, filter_section=None, check_requirements=True):
    """Read and parse the configuration file for RAMP.

    Parameters
    ----------
    config_file: str
        Path to the YAML configuration file.
    filter_section : None, str, or list of str, default is None
        To restrict the configuration to particular field. By default all the
        configuration file is read. The available sections of the configuration
        is:

        * 'sqlalchemy': contain the information related to the database;
    check_requirements: bool, default is True
        Whether to check that all minimum configuration parameters were read
        from the configuration file.

    Returns
    -------
    config : dict
        Configuration parsed as a dictionary.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # if a single string is given, we will later unpack remove the first layer
    # of the dictionary
    unpack = False
    if filter_section is not None:
        if isinstance(filter_section, str):
            filter_section = [filter_section]
            unpack = True
        for sec in filter_section:
            if sec not in config:
                raise ValueError(
                    'The section "{}" is not in the "{}" file. Got these '
                    'sections instead {}.'
                    .format(sec, os.path.basename(config_file),
                            list(config.keys()))
                )
    config = {key: value
              for key, value in config.items()
              if filter_section is None or key in filter_section}

    if check_requirements:
        for section_name, required_field in REQUIRED_KEYS.items():
            if section_name in config:
                missing_parameters = required_field.difference(
                    config[section_name])
                if missing_parameters:
                    raise ValueError(
                        'The section "{}" in the "{}" file is missing the '
                        'required parameters {}.'
                        .format(section_name, os.path.basename(config_file),
                                missing_parameters)
                    )

    if unpack:
        return config[filter_section[0]]
    return config
