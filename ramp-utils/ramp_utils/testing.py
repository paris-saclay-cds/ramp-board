import os

HERE = os.path.dirname(__file__)


def database_config_template():
    """Return the path a template database configuration file.

    Returns
    -------
    filename : str
        The database configuration filename.
    """
    return os.path.join(HERE, 'template', 'database_config.yml')


def ramp_config_template():
    """Return the path a template RAMP configuration file.

    Returns
    -------
    filename : str
        The RAMP configuration filename.
    """
    return os.path.join(HERE, 'template', 'ramp_config.yml')


def ramp_aws_config_template():
    """Return the path a template RAMP configuration AWS file.

    Returns
    -------
    filename : str
        The RAMP configuration on AWS filename.
    """
    return os.path.join(HERE, 'template', 'ramp_config_aws.yml')
