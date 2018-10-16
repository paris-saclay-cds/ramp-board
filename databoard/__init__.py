import os
import logging

from flask import Flask
from flask_mail import Mail
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from distutils.util import strtobool

__version__ = '0.1.dev'


app = Flask('databoard')
app.config.from_object('databoard.config.Config')
test_config = os.environ.get('DATABOARD_TEST')
if test_config is not None:
    if strtobool(test_config):
        app.config.from_object('databoard.config.TestingConfig')
app.debug = False
db = SQLAlchemy(app)
mail = Mail(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in or sign up to access this page.'

logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s',
    filename=app.config['LOG_FILENAME'])
# get rid of annoying skimage debug messages
logging.getLogger('PIL.PngImagePlugin').disabled = True

from databoard import views, model  # noqa
