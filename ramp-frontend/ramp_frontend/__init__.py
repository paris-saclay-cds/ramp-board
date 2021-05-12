from logging.config import dictConfig
import os

from flask import Flask
from flask_login import LoginManager
from flask_mail import Mail
from flask_sqlalchemy import SQLAlchemy

from ramp_database.model import Model

from ._version import __version__  # noqa

all = [
    '__version__'
]

HERE = os.path.dirname(__file__)
db = SQLAlchemy(model_class=Model)
login_manager = LoginManager()
mail = Mail()


def create_app(config):
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

    if "LOGGER" in config:
        logger_config = config.pop("LOGGER")
        dictConfig(logger_config)
    else:
        dictConfig({
            'version': 1,
            'formatters': {'default': {
                'format': '[%(asctime)s] [%(levelname)s] %(message)s',
            }},
            'handlers': {'wsgi': {
                'class': 'logging.StreamHandler',
                'stream': 'ext://flask.logging.wsgi_errors_stream',
                'formatter': 'default'
            }},
            'root': {
                'level': 'INFO',
                'handlers': ['wsgi']
            }
        })

    app = Flask('ramp-frontend', root_path=HERE)
    app.config.update(config)

    with app.app_context():
        db.init_app(app)
        # register the login manager
        login_manager.init_app(app)
        login_manager.login_view = 'auth.login'
        login_manager.login_message = ('Please log in or sign up to access '
                                       'this page.')
        # register the email manager
        mail.init_app(app)
        # register our blueprint
        from .views import admin
        from .views import auth
        from .views import general
        from .views import leaderboard
        from .views import ramp
        app.register_blueprint(admin.mod)
        app.register_blueprint(auth.mod)
        app.register_blueprint(general.mod)
        app.register_blueprint(leaderboard.mod)
        app.register_blueprint(ramp.mod)

        # initialize the database
        db.create_all()
    return app
