import os
import logging

from flask import Flask
from flask_mail import Mail
from flask.ext.login import LoginManager
from flask.ext.sqlalchemy import SQLAlchemy

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


#toaddrs=app.config['MAIL_RECIPIENTS'],
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

from databoard import views, model  # noqa


# Celery conf
from celery import Celery
app.config['CELERY_BROKER_URL'] = 'redis://localhost:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://localhost:6379/0'
app.config['CELERY_TIMEZONE'] = 'UTC'

# get trained tested submission from datarun
from celery.schedules import crontab
app.config['CELERYBEAT_SCHEDULE'] = {
    'get-trained-tested-submission-datarun': {
        'task': 'tasks.get_trained_tested_submission_datarun',
        'schedule': crontab(minute='*/2')
    }
}

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)

datarun_host_url = app.config['DATARUN_URL']
datarun_username = app.config['DATARUN_USERNAME']
datarun_userpassd = app.config['DATARUN_PASSWORD']


@celery.task(name='tasks.get_trained_tested_submission_datarun')
def get_trained_tested_submission_datarun():
    from databoard.db_tools import get_trained_tested_submissions_datarun
    from databoard.db_tools import get_submissions_of_state
    from databoard.db_tools import compute_contributivity
    from databoard.db_tools import compute_historical_contributivity
    submissions = get_submissions_of_state('new')
    list_events = []
    for submission in submissions:
        list_events.append(submission.event.name)
    get_trained_tested_submissions_datarun(submissions, datarun_host_url,
                                           datarun_username, datarun_userpassd)
    for event in list_events:
        compute_contributivity(event_name=event)
        compute_historical_contributivity(event_name=event)
