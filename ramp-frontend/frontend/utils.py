import logging

from flask_mail import Message

from frontend import mail

logger = logging.getLogger('RAMP-FRONTEND')


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
    try:
        msg = Message(subject)
        msg.body = body
        msg.add_recipient(to)
        mail.send(msg)
    except Exception as e:
        logger.error('Mailing error: {}'.format(e))
