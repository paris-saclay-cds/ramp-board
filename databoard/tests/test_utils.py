# coding=utf-8
import pandas as pd
from databoard import utils


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
