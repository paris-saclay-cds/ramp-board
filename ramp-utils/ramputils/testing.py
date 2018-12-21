import os


def path_config_example():
    """Give the path a ``config.yml`` which can be used as an example."""
    module_path = os.path.dirname(__file__)
    return os.path.join(module_path, 'tests', 'data', 'config.yml')