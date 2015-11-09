from sqlalchemy.ext.declarative import declarative_base

from databoard.config import get_session, get_engine
# so set engine (call config.set_engine_and_session) before importing model
engine = get_engine()
session = get_session()


# These should go in config, later into the ramps table
opening_timestamp = None
public_opening_timestamp = None  # before teams can see only their own scores
closing_timestamp = None


DBBase = declarative_base()


# Used both in teams and users
class NameClashError(Exception):
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


def date_time_format(date_time):
    return date_time.strftime('%d%m%Y')