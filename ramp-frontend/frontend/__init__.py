import os

from flask import Flask
from flask_login import LoginManager
from flask_mail import Mail
from flask_sqlalchemy import SQLAlchemy

from .views import general

HERE = os.path.dirname(__file__)
db = SQLAlchemy()


def create_app(config):
    app = Flask('ramp-frontend', root_path=HERE)
    app.config.update(config)
    db.init_app(app)
    # register the login manager
    # login_manager = LoginManager()
    # login_manager.init_app(app)
    # login_manager.login_view = 'login'
    # login_manager.login_message = ('Please log in or sign up to access this '
    #                                'page.')
    # register the email manager
    # mail = Mail(app)
    # register our blueprint
    app.register_blueprint(general.mod)
    return app
