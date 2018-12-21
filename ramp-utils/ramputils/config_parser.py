import yaml


# TODO: add a parameter to enforce checking for missing parameters.
def read_config(config_file):
    """Read and parse the configuration file for RAMP.

    Parameters
    ----------
    config_file: str
        Path to the YAML configuration file.
    check_required : bool, default is True

    Returns
    -------
    config : dict
        Configuration parsed as a dictionary.
    """
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    return config
