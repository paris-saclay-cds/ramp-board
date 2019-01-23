import os

HERE = os.path.dirname(__file__)


def path_config_example():
    """Give the path a ``config.yml`` which can be used as an example."""
    return os.path.join(HERE, 'tests', 'data', 'config.yml')


def flask_config_template():
    """Return the path a template Flask configuration file.

    Returns
    -------
    filename : str
        The Flask configuration filename.
    """
    return os.path.join(HERE, 'template', 'flask_config.yml')


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
