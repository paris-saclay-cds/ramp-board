import os
import logging

from flask import Flask
from flask_mail import Mail
from flask_login import LoginManager

from rampdb.model.base import Model

from .database import SQLAlchemy

__version__ = '0.1.dev'


app = Flask('databoard')

# Load default main config
app_stage = os.getenv('DATABOARD_STAGE', 'DEVELOPMENT').upper()
if app_stage in ['PROD', 'PRODUCTION']:
    app.config.from_object('databoard.default_config.ProductionConfig')
elif app_stage in ['TEST', 'TESTING']:
    app.config.from_object('databoard.default_config.TestingConfig')
elif app_stage in ['DEV', 'DEVELOP', 'DEVELOPMENT']:
    app.config.from_object('databoard.default_config.DevelopmentConfig')
else:
    msg = (
        "Unknown databoard stage: {}"
        "Please set the environment variable `DATABOARD_STAGE` to one of the "
        "available stages : 'TESTING', 'DEVELOPMENT' or 'PRODUCTION'"
    )
    raise AttributeError(msg.format(app_stage))

# Load default database config
app.config.from_object('databoard.default_config.DBConfig')

# Load default internal config
app.config.from_object('databoard.default_config.RampConfig')

# Load user config
user_config = os.getenv('DATABOARD_CONFIG')
if user_config is not None:
    app.config.from_json(user_config)

db = SQLAlchemy(app, Model=Model)
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

####################################################################

ramp_config = app.config.get_namespace('RAMP_')

deployment_path = app.config.get('DEPLOYMENT_PATH')
ramp_kits_path = os.path.join(deployment_path, ramp_config['kits_dir'])
ramp_data_path = os.path.join(deployment_path, ramp_config['data_dir'])

from . import views  # noqa
from . import model  # noqa
from . import db_tools  # noqa
