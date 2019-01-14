# coding=utf-8
import os
import sys
import pandas as pd

from databoard.forms import UserUpdateProfileForm

from databoard import utils
from databoard.utils import encode_string

PYTHON3 = sys.version_info[0] == 3

def test_password_hashing():
    plain_text_password = "hjst3789ep;ocikaqjw"
    hashed_password = utils.get_hashed_password(plain_text_password)
    assert utils.check_password(plain_text_password, hashed_password)
    assert not utils.check_password("hjst3789ep;ocikaqji", hashed_password)


def test_remove_non_ascii():
    assert utils.remove_non_ascii('bla') == 'bla'
    assert utils.remove_non_ascii('blá') == 'bla'


def test_date_time_format():
    assert utils.date_time_format(pd.to_datetime('2018-01-01')) ==\
        '2018-01-01 00:00:00 Mon'


def test_generate_passwords():
    utils.generate_single_password(mywords=None)
    users_to_add_f_name = '/tmp/users.csv'
    password_f_name = '/tmp/users.csv.w_pwd'
    users = pd.DataFrame({'name': ['n1', 'n2']})
    users.to_csv(users_to_add_f_name)
    utils.generate_passwords(users_to_add_f_name, password_f_name)
    os.remove(users_to_add_f_name)
    os.remove(password_f_name)


def test_encode_string():
    if PYTHON3:
        string = encode_string('a string')
        assert isinstance(string, bytes)
        string = encode_string(b'a string')
        assert isinstance(string, bytes)
    else:
        string = encode_string('a string')
        assert isinstance(string, bytes)
