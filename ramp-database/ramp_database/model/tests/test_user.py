import re
import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import EventAdmin
from ramp_database.model import Model
from ramp_database.model import SubmissionSimilarity
from ramp_database.model import Team

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.user import get_user_by_name


@pytest.fixture(scope='module')
def session_scope_module(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_user_model_properties(session_scope_module):
    user = get_user_by_name(session_scope_module, 'test_user')

    assert user.is_active is True
    assert user.is_anonymous is False
    assert user.get_id() == '1'
    assert re.match(r'User\(.*test_user.*\)', str(user))
    assert re.match(r'User\(name=.*test_user.*, lastname.*\)', repr(user))


@pytest.mark.parametrize(
    'backref, expected_type',
    [('admined_events', EventAdmin),
     ('submission_similaritys', SubmissionSimilarity),
     ('admined_teams', Team)]
)
def test_user_model_backref(session_scope_module, backref, expected_type):
    user = get_user_by_name(session_scope_module, 'test_user')
    backref_attr = getattr(user, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)
