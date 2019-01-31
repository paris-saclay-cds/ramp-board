import shutil

import pytest

from rampdb.model import Model
from rampdb.testing import add_users
from rampdb.tools.team import sign_up_team
from rampdb.tools.submission import get_submissions
from rampdb.tools.submission import submit_starting_kits
from rampdb.utils import session_scope
from rampdb.utils import setup_db

from rampdb.tools.event import get_problem

from rampbkd.local import CondaEnvWorker
from rampbkd.dispatcher import Dispatcher

from ramputils import generate_ramp_config
from ramputils import read_config
from ramputils.testing import database_config_template
from ramputils.testing import ramp_config_template

from ramputils.deploy import deploy_ramp_event


@pytest.fixture
def session_scope_function():
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    try:
        yield
    finally:
        shutil.rmtree(
            ramp_config['ramp']['deployment_dir'], ignore_errors=True
        )
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_deploy_ramp_event_options(session_scope_function):
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    ramp_config = generate_ramp_config(ramp_config)
    deploy_ramp_event(database_config_template(), ramp_config_template())
    # deploy again by forcing the deployment
    deploy_ramp_event(
        database_config_template(), ramp_config_template(), force=True
    )
    # do not deploy the kit to trigger the error in the problem with we don't
    # force the deployment
    msg_err = 'The RAMP problem already exists in the database.'
    with pytest.raises(ValueError, match=msg_err):
        with session_scope(database_config['sqlalchemy']) as session:
            problem = get_problem(session, 'iris')
            problem.path_ramp_kits = problem.path_ramp_kits + '_xxx'
            session.commit()
            deploy_ramp_event(
                database_config_template(), ramp_config_template(),
                setup_ramp_repo=False, force=False
            )

            problem = get_problem(session, 'iris')
            problem.path_ramp_kits = ramp_config['ramp_kits_dir']
            problem.path_ramp_data = problem.path_ramp_data + '_xxx'
            session.commit()
            deploy_ramp_event(
                database_config_template(), ramp_config_template(),
                setup_ramp_repo=False, force=False
            )


def test_deploy_ramp_event(session_scope_function):
    database_config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())
    ramp_config = generate_ramp_config(event_config)
    deploy_ramp_event(database_config_template(), ramp_config_template())

    # simulate that we add users and sign-up for the event and that they
    # submitted the starting kit
    with session_scope(database_config['sqlalchemy']) as session:
        add_users(session)
        sign_up_team(session, ramp_config['event_name'], 'test_user')
        submit_starting_kits(session, ramp_config['event_name'], 'test_user',
                             ramp_config['ramp_kit_submissions_dir'])

    # run the dispatcher on the event which are in the dataset
    dispatcher = Dispatcher(config=database_config,
                            event_config=event_config,
                            worker=CondaEnvWorker, n_worker=-1,
                            hunger_policy='exit')
    dispatcher.launch()

    # the iris kit contain a submission which should fail for a user
    with session_scope(database_config['sqlalchemy']) as session:
        submission = get_submissions(
            session, event_config['ramp']['event_name'], 'training_error'
        )
        assert len(submission) == 1
