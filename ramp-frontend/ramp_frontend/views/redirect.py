import logging

from flask import flash
from flask import redirect
from flask import url_for

logger = logging.getLogger('RAMP-FRONTEND')


def redirect_to_user(message_str, is_error=True, category='message'):
    """Redirect the page to the problem landing page.

    Parameters
    ----------
    message_str : str
        The message to inform the user what is going on.
    is_error : bool, default is True
        Whether this is due to an error.
    category : str or None
        The category of the error. Refer to :func:`flask.flash`.
    """
    flash(message_str, category=category)
    print(message_str)
    if is_error:
        logger.error(message_str)
    else:
        logger.info(message_str)
    return redirect(url_for('ramp.problems'))


def redirect_to_sandbox(event, message_str, is_error=True, category=None):
    """Redirect the page to the sandbox landing page.

    Parameters
    ----------
    message_str : str
        The message to inform the user what is going on.
    is_error : bool, default is True
        Whether this is due to an error.
    category : str or None
        The category of the error. Refer to :func:`flask.flash`.
    """
    flash(message_str, category=category)
    if is_error:
        logger.error(message_str)
    else:
        logger.info(message_str)
    return redirect('/events/{}/sandbox'.format(event.name))


def redirect_to_credit(submission_hash, message_str, is_error=True,
                       category=None):
    """Redirect the page to the credit landing page.

    Parameters
    ----------
    submission_hash : str
        The hash of the current submission.
    message_str : str
        The message to inform the user what is going on.
    is_error : bool, default is True
        Whether this is due to an error.
    category : str or None
        The category of the error. Refer to :func:`flask.flash`.
    """
    flash(message_str, category=category)
    if is_error:
        logger.error(message_str)
    else:
        logger.info(message_str)
    return redirect('/credit/{}'.format(submission_hash))
