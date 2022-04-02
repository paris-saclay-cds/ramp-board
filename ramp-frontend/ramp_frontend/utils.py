"""
The :mod:`ramp_frontend.utils` provides utilities to ease sending email.
"""

import logging
from concurrent.futures import ThreadPoolExecutor

from flask import (
    copy_current_request_context,
    current_app,
)
from flask_mail import Message

from ramp_frontend import mail

logger = logging.getLogger("RAMP-FRONTEND")


def ensure_threadpoolexecutor_is_running(pool):
    """Ensure that the threadpool executor is running.

    Parameters
    ----------
    pool : :class:`concurrent.futures.ThreadPoolExecutor`
        The threadpool executor.
    """
    try:
        pool.submit(lambda: None)
        return pool
    except RuntimeError:
        logger.info("Threadpool executor is not running")
        pool.shutdown(wait=False)
        return ThreadPoolExecutor(max_workers=pool._max_workers)


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

    """.format(
        user.name,
        user.firstname,
        user.lastname,
        user.email,
        user.linkedin_url,
        user.twitter_url,
        user.facebook_url,
        user.github_url,
        user.hidden_notes,
        user.bio,
    )

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
    msg = Message(subject)
    msg.body = body
    msg.add_recipient(to)

    @copy_current_request_context
    def send(msg):
        try:
            mail.send(msg)
        except Exception as e:
            logger.error(f"Mailing error: {e}")
        return "Successfully sent email"

    current_app.pool = ensure_threadpoolexecutor_is_running(current_app.pool)
    current_app.pool.submit(send, msg)
