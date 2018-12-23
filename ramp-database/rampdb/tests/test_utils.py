import shutil

import pytest

# TODO: we temporary use the setup of databoard to create a dataset
from databoard import db
from databoard import deployment_path
from databoard.testing import create_test_db

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import SubmissionFileType

from rampdb.utils import setup_db


@pytest.fixture
def database():
    try:
        create_test_db()
        yield
    finally:
        shutil.rmtree(deployment_path, ignore_errors=True)
        db.session.close()
        db.session.remove()
        db.drop_all()


def test_setup_db(database):
    config = read_config(path_config_example(), filter_section='sqlalchemy')
    db, Session = setup_db(config)
    with db.connect() as conn:
        session = Session(bind=conn)
        file_type = session.query(SubmissionFileType).all()
        assert len(file_type) > 0