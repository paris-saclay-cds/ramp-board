import re
import shutil

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Model

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.user import get_user_by_name


@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture(scope='module')
def session_scope_module(config):
    try:
        create_toy_db(config)
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


def test_user_model_properties(session_scope_module):
    user = get_user_by_name(session_scope_module, 'test_user')

    assert user.is_active is True
    assert user.is_anonymous is False
    assert user.get_id() == '1'
    assert re.match(r'User\(.*test_user.*\)', str(user))
    assert re.match(r'User\(name=.*test_user.*, lastname.*\)', repr(user))
