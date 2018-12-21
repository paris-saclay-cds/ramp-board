import importlib
import logging
import sys

import pandas as pd

import bcrypt
from flask_mail import Message
from unidecode import unidecode

from . import mail

logger = logging.getLogger('databoard')

PYTHON3 = sys.version_info[0] == 3


def encode_string(text):
    """Encode text into an array of bytes in both Python 2 and 3 with UTF-8.

    Parameters
    ----------
    text : str or bytes
        The text to be encoded

    Returns
    -------
    encoded_text : bytes
        The encoded text.
    """
    if PYTHON3:
        return bytes(text, 'utf-8') if isinstance(text, str) else text
    return text.encode('utf8')


def remove_non_ascii(text):
    if PYTHON3:
        return unidecode(text)
    return unicode(unidecode(unicode(text, encoding='utf-8')), 'utf-8')


def date_time_format(date_time):
    return date_time.strftime('%Y-%m-%d %H:%M:%S %a')


def table_format(table_html):
    """Remove <table></table> keywords from html table.

    (converted from pandas dataframe), to insert in datatable.
    """
    # return '<thead> %s </tbody><tfoot><tr></tr></tfoot>' %\
    return '<thead> %s </tbody>' %\
        table_html.split('<thead>')[1].split('</tbody>')[0]


def get_hashed_password(plain_text_password):
    """Hash a password for the first time.

    (Using bcrypt, the salt is saved into the hash itself)
    """
    password = encode_string(plain_text_password)

    return bcrypt.hashpw(password, bcrypt.gensalt())


def check_password(plain_text_password, hashed_password):
    """Check hashed password.

    Using bcrypt, the salt is saved into the hash itself.
    """
    password = encode_string(plain_text_password)
    hashed_p = encode_string(hashed_password)

    return bcrypt.checkpw(password, hashed_p)


def generate_single_password(mywords=None):
    import xkcdpass.xkcd_password as xp
    if mywords is None:
        words = xp.locate_wordfile()
        mywords = xp.generate_wordlist(
            wordfile=words, min_length=4, max_length=6)
    return xp.generate_xkcdpassword(mywords, numwords=4)


def generate_passwords(users_to_add_f_name, password_f_name):
    import xkcdpass.xkcd_password as xp
    users_to_add = pd.read_csv(users_to_add_f_name)
    words = xp.locate_wordfile()
    mywords = xp.generate_wordlist(wordfile=words, min_length=4, max_length=6)
    users_to_add['password'] = [
        generate_single_password(mywords) for name in users_to_add['name']]
    # temporarily while we don't implement pwd recovery
    users_to_add[['name', 'password']].to_csv(password_f_name, index=False)


def send_mail(to, subject, body):
    try:
        # Create message
        msg = Message(subject)
        msg.body = body
        msg.add_recipient(to)
        # Send email
        mail.send(msg)
    except Exception as e:
        logger.error('Mailing error: {}'.format(e))

# def send_mail(to, subject, body):
#     try:
#         logger.info('Sending "{}" mail to {}'.format(subject, to))
#         sender_user = config.MAIL_USERNAME
#         sender_pwd = config.MAIL_PASSWORD
#         smtpserver = smtplib.SMTP(config.MAIL_SERVER, config.MAIL_PORT)
#         smtpserver.ehlo()
#         smtpserver.starttls()
#         smtpserver.ehlo
#         smtpserver.login(sender_user, sender_pwd)
#         header = 'To: {}\nFrom: RAMP admin <{}>\nSubject: {}\n\n'.format(
#             to, sender_user, subject.encode('utf-8'))
#         smtpserver.sendmail(sender_user, to, header + body)
#     except Exception as e:
#         logger.error('Mailing error: {}'.format(e))
