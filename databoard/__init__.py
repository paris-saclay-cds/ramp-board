import logging

from flask import Flask
from logging.handlers import SMTPHandler

app = Flask(__name__)
import databoard.views

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_DEBUG'] = app.debug
app.config['MAIL_USERNAME'] = 'databoardmailer@gmail.com'
app.config['MAIL_PASSWORD'] = 'peace27man'
app.config['MAIL_DEFAULT_SENDER'] = ('Databoard', 'databoardmailer@gmail.com')
app.config['MAIL_RECIPIENTS'] = '' #notification_recipients


if app.debug:
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(levelname)s: %(message)s',
        filename=app.config['LOG_FILENAME'])
else:
    mail_handler = SMTPHandler(mailhost=app.config['MAIL_SERVER'],
        fromaddr=app.config['MAIL_DEFAULT_SENDER'],
        toaddrs=app.config['MAIL_RECIPIENTS'], 
        subject='Databoard error')
    mail_handler.setFormatter(logging.Formatter('''\
        Message type:       %(levelname)s
        Location:           %(pathname)s:%(lineno)d
        Module:             %(module)s
        Function:           %(funcName)s
        Time:               %(asctime)s
        Message:
        %(message)s
        '''))

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.ERROR)
    logger.addHandler(mail_handler)

