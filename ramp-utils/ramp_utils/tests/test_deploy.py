import os
import shutil

import pytest

from ramp_database.model import Model
from ramp_database.testing import add_users
from ramp_database.tools.team import sign_up_team
from ramp_database.tools.submission import get_submissions
from ramp_database.tools.submission import submit_starting_kits
from ramp_database.utils import session_scope
from ramp_database.utils import setup_db

from ramp_database.tools.event import get_problem

from ramp_engine.local import CondaEnvWorker
from ramp_engine.dispatcher import Dispatcher

from ramp_utils import generate_ramp_config
from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_utils.deploy import deploy_ramp_event


@pytest.fixture
def session_scope_function(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    try:
        yield
    finally:
        # FIXME: we are recreating the deployment directory but it should be
        # replaced by an temporary creation of folder.
        deployment_dir = os.path.commonpath(
            [ramp_config['ramp']['kit_dir'], ramp_config['ramp']['data_dir']]
        )
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_deploy_ramp_event_options(session_scope_function):
    database_config = read_config(database_config_template())
    ramp_config = generate_ramp_config(read_config(ramp_config_template()))
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
            # if one of the ramp-kit or ramp-data folders changed
            problem = get_problem(session, 'iris')
            problem.path_ramp_kit = problem.path_ramp_kit + '_xxx'
            session.commit()
            deploy_ramp_event(
                database_config_template(), ramp_config_template(),
                setup_ramp_repo=False, force=False
            )

            problem = get_problem(session, 'iris')
            problem.path_ramp_kit = ramp_config['ramp_kit_dir']
            problem.path_ramp_data = problem.path_ramp_data + '_xxx'
            session.commit()
            deploy_ramp_event(
                database_config_template(), ramp_config_template(),
                setup_ramp_repo=False, force=False
            )

    msg_err = 'Attempting to overwrite existing event'
    with pytest.raises(ValueError, match=msg_err):
        with session_scope(database_config['sqlalchemy']) as session:
            # if the problem is the same, then the event should be overwritten
            problem = get_problem(session, 'iris')
            problem.path_ramp_kit = ramp_config['ramp_kit_dir']
            problem.path_ramp_data = ramp_config['ramp_data_dir']
            session.commit()
            deploy_ramp_event(
                database_config_template(), ramp_config_template(),
                setup_ramp_repo=False, force=False
            )


def test_deploy_ramp_event(session_scope_function):
    database_config = read_config(database_config_template())
    event_config_filename = ramp_config_template()
    event_config = read_config(event_config_filename)
    ramp_config = generate_ramp_config(event_config)
    deploy_ramp_event(database_config_template(), ramp_config_template())

    # check that we created the archive
    assert os.path.isfile(
        os.path.join(
            ramp_config['ramp_kit_dir'], 'events_archived',
            ramp_config['event_name'] + '.zip'
        )
    )

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
                            worker=CondaEnvWorker,
                            n_workers=-1,
                            hunger_policy='exit')
    dispatcher.launch()

    # the iris kit contain a submission which should fail for a user
    with session_scope(database_config['sqlalchemy']) as session:
        submission = get_submissions(
            session, event_config['ramp']['event_name'], 'training_error'
        )
        assert len(submission) == 1
