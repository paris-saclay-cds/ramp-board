import shutil

import pytest

from ramputils import read_config
from ramputils.testing import database_config_template
from ramputils.testing import ramp_config_template

from rampdb.model import Model
from rampdb.utils import setup_db
from rampdb.utils import session_scope
from rampdb.testing import create_toy_db

from rampdb.tools.event import get_event
from rampdb.tools.submission import get_submissions

from rampbkd.local import CondaEnvWorker
from rampbkd.dispatcher import Dispatcher


@pytest.fixture
def session_toy():
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


def test_integration_dispatcher(session_toy):
    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())
    dispatcher = Dispatcher(
        config=config, event_config=event_config, worker=CondaEnvWorker,
        n_worker=-1, hunger_policy='exit'
    )
    dispatcher.launch()

    # the iris kit contain a submission which should fail for each user
    submission = get_submissions(
        session_toy, event_config['ramp']['event_name'], 'training_error'
    )
    assert len(submission) == 2


def test_unit_test_dispatcher(session_toy):
    # make sure that the size of the list is bigger than the number of
    # submissions
    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())
    dispatcher = Dispatcher(config=config,
                            event_config=event_config,
                            worker=CondaEnvWorker, n_worker=100,
                            hunger_policy='exit')

    # check that all the queue are empty
    assert dispatcher._awaiting_worker_queue.empty()
    assert dispatcher._processing_worker_queue.empty()
    assert dispatcher._processed_submission_queue.empty()

    # check that all submissions are queued
    submissions = get_submissions(session_toy, 'iris_test', 'new')
    dispatcher.fetch_from_db(session_toy)
    # we should remove the starting kit from the length of the submissions for
    # each user
    assert dispatcher._awaiting_worker_queue.qsize() == len(submissions) - 2
    submissions = get_submissions(session_toy, 'iris_test', 'sent_to_training')
    assert len(submissions) == 6

    # start the training
    dispatcher.launch_workers(session_toy)
    # be sure that the training is finished
    while not dispatcher._processing_worker_queue.empty():
        dispatcher.collect_result(session_toy)

    assert len(get_submissions(session_toy, 'iris_test', 'new')) == 2
    assert (len(get_submissions(session_toy, 'iris_test', 'training_error')) ==
            2)
    assert len(get_submissions(session_toy, 'iris_test', 'tested')) == 4

    dispatcher.update_database_results(session_toy)
    assert dispatcher._processed_submission_queue.empty()
    event = get_event(session_toy, 'iris_test')
    assert event.private_leaderboard_html
    assert event.public_leaderboard_html_with_links
    assert event.public_leaderboard_html_no_links
    assert event.failed_leaderboard_html
    assert event.new_leaderboard_html is None
    assert event.public_competition_leaderboard_html
    assert event.private_competition_leaderboard_html
