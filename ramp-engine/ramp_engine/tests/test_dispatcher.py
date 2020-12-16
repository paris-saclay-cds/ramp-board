import os
import pytest
import shutil
from unittest import mock

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_aws_config_template, ramp_config_template

from ramp_database.model import Model
from ramp_database.utils import setup_db
from ramp_database.utils import session_scope
from ramp_database.testing import create_toy_db

from ramp_database.tools.event import get_event
from ramp_database.tools.submission import get_submissions
from ramp_database.tools.submission import get_submission_by_id

from ramp_engine.local import CondaEnvWorker
from ramp_engine.aws import AWSWorker
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


@pytest.fixture
def session_toy_aws(database_connection):
    database_config = read_config(database_config_template())
    ramp_config_aws = ramp_aws_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config_aws)
        with session_scope(database_config['sqlalchemy']) as session:
            yield session
    finally:
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)
        shutil.rmtree(deployment_dir, ignore_errors=True)


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


def test_dispatcher_aws_not_launching(session_toy_aws, caplog):
    # given the test config file the instance should not be able to launch
    # due to authentication error
    # after unsuccessful try the worker should teardown
    config = read_config(database_config_template())
    event_config = read_config(ramp_aws_config_template())

    dispatcher = Dispatcher(config=config,
                            event_config=event_config,
                            worker=AWSWorker, n_workers=10,
                            hunger_policy='exit')
    dispatcher.fetch_from_db(session_toy_aws)
    submissions = get_submissions(session_toy_aws, 'iris_aws_test', 'new')

    dispatcher.launch_workers(session_toy_aws)
    assert 'AuthFailure' in caplog.text
    # training should not have started
    assert 'training' not in caplog.text
    num_running_workers = dispatcher._processing_worker_queue.qsize()
    assert num_running_workers == 0
    submissions2 = get_submissions(session_toy_aws, 'iris_aws_test', 'new')
    # assert that all the submissions are still in the 'new' state
    assert len(submissions) == len(submissions2)


@mock.patch('ramp_engine.aws.api.download_log')
@mock.patch('ramp_engine.aws.api.check_instance_status')
@mock.patch('ramp_engine.aws.api._get_log_content')
@mock.patch('ramp_engine.aws.api._training_successful')
@mock.patch('ramp_engine.aws.api._training_finished')
@mock.patch('ramp_engine.aws.api.is_spot_terminated')
@mock.patch('ramp_engine.aws.api.launch_train')
@mock.patch('ramp_engine.aws.api.upload_submission')
@mock.patch('ramp_engine.aws.api.launch_ec2_instances')
def test_info_on_training_error(test_launch_ec2_instances, upload_submission,
                                launch_train,
                                is_spot_terminated, training_finished,
                                training_successful,
                                get_log_content, check_instance_status,
                                download_log,
                                session_toy_aws,
                                caplog):
    # make sure that the Python error from the solution is passed to the
    # dispatcher
    # everything shoud be mocked as correct output from AWS instances
    # on setting up the instance and loading the submission
    # mock dummy AWS instance
    class DummyInstance:
        id = 1
    test_launch_ec2_instances.return_value = (DummyInstance(),), 0
    upload_submission.return_value = 0
    launch_train.return_value = 0
    is_spot_terminated.return_value = 0
    training_finished.return_value = False
    download_log.return_value = 0

    config = read_config(database_config_template())
    event_config = read_config(ramp_aws_config_template())

    dispatcher = Dispatcher(config=config,
                            event_config=event_config,
                            worker=AWSWorker, n_workers=10,
                            hunger_policy='exit')
    dispatcher.fetch_from_db(session_toy_aws)
    dispatcher.launch_workers(session_toy_aws)
    num_running_workers = dispatcher._processing_worker_queue.qsize()
    # worker, (submission_id, submission_name) = \
    #     dispatcher._processing_worker_queue.get()
    # assert worker.status == 'running'
    submissions = get_submissions(session_toy_aws,
                                  'iris_aws_test',
                                  'training')
    ids = [submissions[idx][0] for idx in range(len(submissions))]
    assert len(submissions) > 1
    assert num_running_workers == len(ids)

    dispatcher.time_between_collection = 0
    training_successful.return_value = False

    # now we will end the submission with training error
    training_finished.return_value = True
    training_error_msg = 'Python error here'
    get_log_content.return_value = training_error_msg
    check_instance_status.return_value = 'finished'

    dispatcher.collect_result(session_toy_aws)

    # the worker which we were using should have been teared down
    num_running_workers = dispatcher._processing_worker_queue.qsize()

    assert num_running_workers == 0

    submissions = get_submissions(session_toy_aws,
                                  'iris_aws_test',
                                  'training_error')
    assert len(submissions) == len(ids)

    submission = get_submission_by_id(session_toy_aws, submissions[0][0])
    assert training_error_msg in submission.error_msg
