import os
import logging

from flask import Flask
from flask_mail import Mail
from flask_login import LoginManager
from flask_sqlalchemy import SQLAlchemy
from flask_wtf.csrf import CSRFProtect
from distutils.util import strtobool

__version__ = '0.1.dev'


app = Flask('databoard')
app.config.from_object('databoard.config.Config')
test_config = os.environ.get('DATABOARD_TEST')
if test_config is not None:
    if strtobool(test_config):
        app.config.from_object('databoard.config.TestingConfig')
app.debug = False
# CSRFProtect(app)
# app.debug_mode = True
db = SQLAlchemy(app)
mail = Mail(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'
login_manager.login_message = 'Please log in or sign up to access this page.'

# if app.debug:
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s',
    filename=app.config['LOG_FILENAME'])
# get rid of annoying skimage debug messages
logging.getLogger('PIL.PngImagePlugin').disabled = True
# else:
# toaddrs=app.config['MAIL_RECIPIENTS'],
#         subject='Databoard error')
#     mail_handler.setFormatter(logging.Formatter('''\
#         Message type:       %(levelname)s
#         Location:           %(pathname)s:%(lineno)d
#         Module:             %(module)s
#         Function:           %(funcName)s
#         Time:               %(asctime)s
#         Message:
#         %(message)s
#         '''))

#     logger = logging.getLogger('databoard')
#     logger.setLevel(logging.ERROR)
#     logger.addHandler(mail_handler)

# Celery conf
# app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
# app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
# app.config['CELERY_TIMEZONE'] = 'UTC'
#
# # get trained tested submission from datarun
# app.config['CELERYBEAT_SCHEDULE'] = {
#     'get-trained-tested-submission-datarun': {
#         'task': 'tasks.get_submissions_datarun',
#         'schedule': crontab(minute='*/2')
#     }
# }

from databoard import views, model  # noqa
