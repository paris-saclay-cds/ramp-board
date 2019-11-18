import shutil

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import CVFold
from ramp_database.model import Model
from ramp_database.model import SubmissionOnCVFold

from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.event import get_event


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


def test_cv_fold_model(session_scope_module):
    event = get_event(session_scope_module, 'iris_test')
    cv_fold = (session_scope_module.query(CVFold)
                                   .filter(CVFold.event_id == event.id)
                                   .all())
    assert "train fold [" in repr(cv_fold[0])


@pytest.mark.parametrize(
    'backref, expected_type',
    [('submissions', SubmissionOnCVFold)]
)
def test_cv_fold_model_backref(session_scope_module, backref, expected_type):
    event = get_event(session_scope_module, 'iris_test')
    cv_fold = (session_scope_module.query(CVFold)
                                   .filter(CVFold.event_id == event.id)
                                   .first())
    backref_attr = getattr(cv_fold, backref)
    assert isinstance(backref_attr, list)
    # only check if the list is not empty
    if backref_attr:
        assert isinstance(backref_attr[0], expected_type)
