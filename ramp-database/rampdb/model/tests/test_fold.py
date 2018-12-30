import datetime
import shutil

import pytest

from ramputils import read_config
from ramputils.testing import path_config_example


from rampdb.model import CVFold
from rampdb.model import Model

from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.event import get_event


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


def test_cv_fold_model(session_scope_module):
    event = get_event(session_scope_module, 'iris_test')
    cv_fold = (session_scope_module.query(CVFold)
                                   .filter(CVFold.event_id == event.id)
                                   .all())
    assert  "train fold [" in repr(cv_fold[0])
