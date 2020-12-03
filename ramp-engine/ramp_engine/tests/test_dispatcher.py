import shutil
import os

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Model
from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.event import get_event
from ramp_database.tools.submission import get_submissions
from ramp_database.tools.submission import get_submission_by_id

from ramp_engine.local import CondaEnvWorker
from ramp_engine.dispatcher import Dispatcher


HERE = os.path.dirname(__file__)


@pytest.fixture
def session_toy(database_connection):
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


def test_error_handling_worker_setup_error(session_toy, caplog):
    # make sure the error on the worker.setup is dealt with correctly
    # set mock worker
    class Worker_mock():
        def __init__(self, *args, **kwargs):
            self.state = None

        def setup(self):
            raise Exception('Test error')

        def teardown(self):
            pass

    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())

    worker = Worker_mock()
    dispatcher = Dispatcher(
        config=config, event_config=event_config, worker=Worker_mock,
        n_workers=-1, hunger_policy='exit'
    )

    dispatcher.launch()
    submissions = get_submissions(
        session_toy, event_config['ramp']['event_name'], 'checking_error'
    )
    assert len(submissions) == 6
    worker.status = 'error'
    assert 'Test error' in caplog.text


def test_integration_dispatcher(session_toy):
    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())
    dispatcher = Dispatcher(
        config=config, event_config=event_config, worker=CondaEnvWorker,
        n_workers=-1, hunger_policy='exit'
    )
    dispatcher.launch()

    # the iris kit contain a submission which should fail for each user
    submissions = get_submissions(
        session_toy, event_config['ramp']['event_name'], 'training_error'
    )
    assert len(submissions) == 2
    submission = get_submission_by_id(session_toy, submissions[0][0])
    assert 'ValueError' in submission.error_msg


def test_unit_test_dispatcher(session_toy):
    # make sure that the size of the list is bigger than the number of
    # submissions
    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())
    dispatcher = Dispatcher(config=config,
                            event_config=event_config,
                            worker=CondaEnvWorker, n_workers=100,
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


@pytest.mark.parametrize(
    "n_threads", [None, 4]
)
def test_dispatcher_num_threads(n_threads):
    libraries = ('OMP', 'MKL', 'OPENBLAS')
    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())

    # check that by default we don't set the environment by default
    dispatcher = Dispatcher(config=config,
                            event_config=event_config,
                            worker=CondaEnvWorker, n_workers=100,
                            n_threads=n_threads,
                            hunger_policy='exit')
    if n_threads is None:
        assert dispatcher.n_threads is n_threads
        for lib in libraries:
            assert getattr(os.environ, lib + "_NUM_THREADS", None) is None
    else:
        assert dispatcher.n_threads == n_threads
        for lib in libraries:
            assert os.environ[lib + "_NUM_THREADS"] == str(n_threads)


def test_dispatcher_error():
    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())

    # check that passing a not a number will raise a TypeError
    err_msg = "The parameter 'n_threads' should be a positive integer"
    with pytest.raises(TypeError, match=err_msg):
        Dispatcher(config=config,
                   event_config=event_config,
                   worker=CondaEnvWorker, n_workers=100,
                   n_threads='whatever',
                   hunger_policy='exit')


def test_dispatcher_timeout(session_toy):
    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())
    dispatcher = Dispatcher(
        config=config, event_config=event_config, worker=CondaEnvWorker,
        n_workers=-1, hunger_policy='exit'
    )
    # override the timeout of the worker
    dispatcher._worker_config["timeout"] = 1
    dispatcher.launch()

    # we should have at least 3 submissions which will fail:
    # 2 for errors and 1 for timeout
    submissions = get_submissions(
        session_toy, event_config['ramp']['event_name'], 'training_error'
    )
    assert len(submissions) >= 2


def test_dispatcher_worker_retry(session_toy):
    config = read_config(database_config_template())
    event_config = read_config(ramp_config_template())
    dispatcher = Dispatcher(config=config,
                            event_config=event_config,
                            worker=CondaEnvWorker, n_workers=10,
                            hunger_policy='exit')

    dispatcher.fetch_from_db(session_toy)
    dispatcher.launch_workers(session_toy)

    # Get one worker and set status to 'retry'
    worker, (submission_id, submission_name) = \
        dispatcher._processing_worker_queue.get()
    setattr(worker, 'status', 'retry')
    assert worker.status == 'retry'
    # Add back to queue
    dispatcher._processing_worker_queue.put_nowait(
        (worker, (submission_id, submission_name))
    )

    while not dispatcher._processing_worker_queue.empty():
        dispatcher.collect_result(session_toy)

    submissions = get_submissions(session_toy, 'iris_test', 'new')
    assert submission_name in [sub[1] for sub in submissions]
