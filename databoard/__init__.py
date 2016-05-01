import logging

from flask import Flask
from flask_mail import Mail
from flask.ext.login import LoginManager
from flask.ext.sqlalchemy import SQLAlchemy

__version__ = '0.1.dev'


app = Flask('databoard')
app.config.from_object('databoard.config')
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
