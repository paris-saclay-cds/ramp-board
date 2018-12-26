import shutil

import pytest

from ramputils import read_config
from ramputils import generate_ramp_config
from ramputils.testing import path_config_example

from rampdb.model import Model

from rampdb.utils import setup_db
from rampdb.utils import session_scope

from rampdb.testing import create_test_db
from rampdb.testing import _setup_ramp_kits_ramp_data

from rampdb.tools.event import add_problem

from rampdb.tools.event import get_problem


@pytest.fixture(scope='module')
def database_config():
    return read_config(path_config_example(), filter_section='sqlalchemy')

@pytest.fixture(scope='module')
def config():
    return read_config(path_config_example())


@pytest.fixture
def session_scope_function(config):
    try:
        create_test_db(config)
        with session_scope(config['sqlalchemy']) as session:
            yield session
    finally:
        shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
        db, Session = setup_db(config['sqlalchemy'])
        with db.connect() as conn:
            session = Session(bind=conn)
            session.close()
        Model.metadata.drop_all(db)


def test_add_problem(session_scope_function, config):
    problem_name = 'iris'
    _setup_ramp_kits_ramp_data(config, problem_name)
    ramp_config = generate_ramp_config(config)
    add_problem(session_scope_function, problem_name,
                ramp_config['ramp_kits_dir'])
    problem = get_problem(session_scope_function, problem_name)
    print(problem)
