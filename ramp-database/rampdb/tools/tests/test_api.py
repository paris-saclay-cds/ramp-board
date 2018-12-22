import shutil
import pytest

# TODO: we temporary use the setup of databoard to create a dataset
from databoard import db
from databoard import deployment_path
from databoard.testing import create_toy_db

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.tools import get_event_nb_folds
from rampdb.tools import get_submission_by_id
from rampdb.tools import get_submission_by_name
from rampdb.tools import get_submission_state
from rampdb.tools import get_submissions
from rampdb.tools import set_predictions
from rampdb.tools import set_submission_error_msg
from rampdb.tools import set_submission_max_ram
from rampdb.tools import set_submission_state
from rampdb.tools import score_submission


@pytest.fixture(scope='module')
def config_database():
    return read_config(path_config_example(), filter_section='sqlalchemy')


@pytest.fixture(scope='module')
def setup_db():
    try:
        create_toy_db()
        yield
    finally:
        shutil.rmtree(deployment_path, ignore_errors=True)
        db.session.close()
        db.session.remove()
        db.drop_all()


def test_hot_test(config_database, setup_db):
    print(get_submissions(config_database, 'iris_test'))
    get_submission_by_id(config_database, 7)
    get_submission_by_name(config_database, 'iris_test', 'test_user',
                           'starting_kit_test')
    get_submission_state(config_database, 7)
    get_event_nb_folds(config_database, 'iris_test')
    set_submission_state(config_database, 7, 'trained')
    set_submission_max_ram(config_database, 7, 100)
    set_submission_error_msg(config_database, 7, 'xxxx')
