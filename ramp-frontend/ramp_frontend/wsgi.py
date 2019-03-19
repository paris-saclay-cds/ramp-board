from ramp_utils import generate_flask_config
from ramp_utils import read_config

from ramp_frontend import create_app


def make_app(config_file):
    """Create the RAMP Flask app and register the views.

    Parameters
    ----------
    config : dict
        The Flask configuration generated with
        :func:`ramp_utils.generate_flask_config`.

    Returns
    -------
    app : Flask
        The Flask app created.
    """
    config = read_config(config_file)
    flask_config = generate_flask_config(config)
    app = create_app(flask_config)
    return app
