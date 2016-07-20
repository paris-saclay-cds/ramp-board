import os
import logging

from flask import Flask
from flask_mail import Mail
from flask.ext.login import LoginManager
from flask.ext.sqlalchemy import SQLAlchemy

from celery import Celery
# from celery.schedules import crontab

__version__ = '0.1.dev'


app = Flask('databoard')
app.config.from_object('databoard.config.Config')
if os.environ.get('DATABOARD_TEST'):
    app.config.from_object('databoard.config.TestingConfig')
app.debug = False
# app.debug_mode = True
db = SQLAlchemy(app)
mail = Mail(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = '/'
login_manager.login_message = 'Please log in or sign up to access this page.'

# if app.debug:
logging.basicConfig(
    level=logging.DEBUG,
    format='%(levelname)s: %(message)s',
    filename=app.config['LOG_FILENAME'])
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


def make_celery(app):
    celery = Celery(app.import_name,
                    backend=app.config['CELERY_RESULT_BACKEND'],
                    broker=app.config['CELERY_BROKER_URL'])
    celery.conf.update(app.config)
    TaskBase = celery.Task

    class ContextTask(TaskBase):
        abstract = True

        def __call__(self, *args, **kwargs):
            with app.app_context():
                return TaskBase.__call__(self, *args, **kwargs)

    celery.Task = ContextTask
    return celery

celery = make_celery(app)

from databoard import views, model  # noqa
