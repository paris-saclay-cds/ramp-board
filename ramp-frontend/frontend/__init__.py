import os

from flask import Flask
from flask_login import LoginManager
from flask_mail import Mail
from flask_sqlalchemy import SQLAlchemy

from rampdb.model import Model

from ._version import __version__

all = [
    '__version__'
]

HERE = os.path.dirname(__file__)
db = SQLAlchemy(model_class=Model)
login_manager = LoginManager()
mail = Mail()


def create_app(config):
    app = Flask('ramp-frontend', root_path=HERE)
    app.config.update(config)
    with app.app_context():
        Model.metadata.create_all(db)
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
    return app
