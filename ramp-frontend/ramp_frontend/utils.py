"""
The :mod:`ramp_frontend.utils` provides utilities to ease sending email.
"""

import logging
from functools import wraps
from threading import Thread

from flask import current_app
from flask_mail import Message

from ramp_frontend import mail

logger = logging.getLogger('RAMP-FRONTEND')


def async_task(f):
    """ Takes a function and runs it in a thread """
    @wraps(f)
    def _decorated(*args, **kwargs):
        thr = Thread(target=f, args=args, kwargs=kwargs)
        thr.start()
    return _decorated


def body_formatter_user(user):
    """Create the body of an email using the user information.

    Parameters
    ----------
    user : :class:`ramp_database.model.User`
        The user profile.

    Returns
    -------
    body : str
        The email body.
    """
    body = """
    user = {}
    name = {} {}
    email = {}
    linkedin = {}
    twitter = {}
    facebook = {}
    github = {}
    notes = {}
    bio = {}

    """.format(user.name, user.firstname,
               user.lastname, user.email, user.linkedin_url,
               user.twitter_url, user.facebook_url, user.github_url,
               user.hidden_notes, user.bio)

    return body


def send_mail(to, subject, body):
    """Send email using Flask Mail.

    Parameters
    ----------
    to : str
        The email address of the recipient.
    subject : str
        The subject of the email.
    body : str
        The body of the email.
    """
    app = current_app._get_current_object()
    try:
        msg = Message(subject)
        msg.body = body
        msg.add_recipient(to)
        _send_async_email(app, msg)
    except Exception as e:
        logger.error('Mailing error: {}'.format(e))


@async_task
def _send_async_email(flask_app, msg):
    """ Sends an send_email asynchronously
    Args:
        flask_app (flask.Flask): Current flask instance
        msg (Message): Message to send
    Returns:
        None
    """
    with flask_app.app_context():
        mail.send(msg)
