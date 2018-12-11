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
    if PYTHON3:
        if isinstance(text, str):
            encoded_text = bytes(text, 'utf-8')
        else:
            encoded_text = text
    else:
        encoded_text = text.encode('utf8')

    return encoded_text


def remove_non_ascii(text):
    if PYTHON3:
        return unidecode(text)
    else:
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


def import_module_from_source(source, name):
    """Load a module from a Python source file.

    Parameters
    ----------
    source : str
        Path to the Python source file which will be loaded as a module.
    name : str
        Name to give to the module once loaded.
    Returns
    -------
    module : Python module
        Return the Python module which has been loaded.
    """
    if sys.version_info[0] < 3:
        import imp
        module = imp.load_source(name, source)
        return module
    else:
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
