import datetime
import shutil

import pytest

from ramputils import read_config
from ramputils.testing import database_config_template
from ramputils.testing import ramp_config_template

from rampdb.model import CVFold
from rampdb.model import Model
from rampdb.model import SubmissionOnCVFold

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.event import get_event


@pytest.fixture(scope='module')
def session_scope_module():
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    try:
        create_toy_db(database_config, ramp_config)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(
            ramp_config['ramp']['deployment_dir'], ignore_errors=True
        )
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
