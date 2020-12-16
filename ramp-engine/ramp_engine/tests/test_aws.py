"""
For testing the Amazon worker locally, you need to:
copy the _aws_config.yml file and rename it config.yml, then update it using
the credentials to interact with Amazon

"""
import botocore
import logging
import os
import shutil
import subprocess

import pytest

from ramp_database.tools.submission import get_submissions
from ramp_engine.aws.api import is_spot_terminated, launch_ec2_instances
from ramp_engine.aws.api import download_log, download_predictions
from ramp_engine.aws.api import upload_submission
from ramp_engine import Dispatcher, AWSWorker
from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_aws_config_template, ramp_config_template

from .test_dispatcher import session_toy  # noqa

from unittest import mock

HERE = os.path.dirname(__file__)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def add_empty_dir(dir_name):
    if not os.path.exists(dir_name):
        os.mkdir(dir_name)


@mock.patch('ramp_engine.aws.api.launch_ec2_instances')
def test_launch_ec2_instances_put_back_into_queue(test_launch_ec2_instances,
                                                  caplog):
    ''' checks if the retry status and the correct log is added if the
        api returns None instances and status retry '''

    test_launch_ec2_instances.return_value = None, 'retry'

    # setup the AWS worker
    event_config = read_config(ramp_aws_config_template())['worker']

    worker = AWSWorker(event_config, submission='starting_kit_local')
    worker.config = event_config

    # worker should be put back into the queue
    worker.setup()
    assert worker.status == 'retry'
    assert 'Adding it back to the queue and will try again' in caplog.text


@mock.patch('ramp_engine.aws.api._rsync')
@mock.patch('ramp_engine.aws.api.launch_ec2_instances')
def test_aws_worker_upload_error(test_launch_ec2_instances, test_rsync,
                                 caplog):
    # mock dummy AWS instance
    class DummyInstance:
        id = 1

    test_launch_ec2_instances.return_value = (DummyInstance(),), 0
    # mock the called process error
    test_rsync.side_effect = subprocess.CalledProcessError(255, 'test')

    # setup the AWS worker
    event_config = read_config(ramp_aws_config_template())['worker']

    worker = AWSWorker(event_config, submission='starting_kit_local')
    worker.config = event_config

    # CalledProcessError is thrown inside
    worker.setup()
    assert worker.status == 'error'
    assert 'Unable to connect during log download' in caplog.text


@mock.patch('ramp_engine.aws.api._rsync')
@mock.patch("ramp_engine.base.BaseWorker.collect_results")
def test_aws_worker_download_log_error(superclass, test_rsync,
                                       caplog):
    # mock dummy AWS instance
    class DummyInstance:
        id = 'test'

    test_rsync.side_effect = subprocess.CalledProcessError(255, 'test')

    # setup the AWS worker
    superclass.return_value = True
    event_config = read_config(ramp_aws_config_template())['worker']

    worker = AWSWorker(event_config, submission='starting_kit_local')
    worker.config = event_config
    worker.status = 'finished'
    worker.instance = DummyInstance
    # worker will now through an CalledProcessError
    exit_status, error_msg = worker.collect_results()
    assert 'Error occurred when downloading the logs' in caplog.text
    assert 'Trying to download the log once again' in caplog.text
    assert exit_status == 2
    assert 'test' in error_msg
    assert worker.status == 'error'


@mock.patch('ramp_engine.aws.api._rsync')
@mock.patch('ramp_engine.aws.api._training_successful')
@mock.patch('ramp_engine.aws.api.download_log')
@mock.patch("ramp_engine.base.BaseWorker.collect_results")
def test_aws_worker_download_prediction_error(superclass, test_download_log,
                                              test_train, test_rsync, caplog):
    # mock dummy AWS instance
    class DummyInstance:
        id = 'test'

    test_rsync.side_effect = subprocess.CalledProcessError(255, 'test')

    test_download_log.return_value = (0,)
    # setup the AWS worker
    superclass.return_value = True
    test_train.return_value = True
    event_config = read_config(ramp_aws_config_template())['worker']

    worker = AWSWorker(event_config, submission='starting_kit_local')
    worker.config = event_config
    worker.status = 'finished'
    worker.instance = DummyInstance
    # worker will now through an CalledProcessError
    exit_status, error_msg = worker.collect_results()
    assert 'Downloading the prediction failed with' in caplog.text
    assert 'Trying to download the prediction once again' in caplog.text
    assert exit_status == 1
    assert 'test' in error_msg


@mock.patch('ramp_engine.aws.api._rsync')
def test_rsync_download_predictions(test_rsync, caplog):
    error = subprocess.CalledProcessError(255, 'test')
    event_config = read_config(ramp_aws_config_template())['worker']
    instance_id = 0
    submission_name = 'test_submission'

    # test for 2 errors by rsync followed by a log output
    test_rsync.side_effect = [error, error, 'test_log']
    out = download_predictions(event_config, instance_id,
                               submission_name, folder=None)
    assert 'Trying to download the prediction' in caplog.text
    assert 'test_submission' in out

    # test for 3 errors by rsync followed by a log output
    test_rsync.side_effect = [error, error, error]
    with pytest.raises(subprocess.CalledProcessError):
        out = download_predictions(event_config, instance_id, submission_name,
                                   folder=None)
    assert 'Trying to download the prediction' in caplog.text
    assert 'error occured when downloading prediction' in caplog.text


@mock.patch('ramp_engine.aws.api._rsync')
def test_rsync_download_log(test_rsync, caplog):
    error = subprocess.CalledProcessError(255, 'test')
    event_config = read_config(ramp_aws_config_template())['worker']
    instance_id = 0
    submission_name = 'test_submission'

    # test for 2 errors by rsync followed by a log output
    test_rsync.side_effect = [error, error, 'test_log']
    out = download_log(event_config, instance_id, submission_name)
    assert 'Trying to download the log' in caplog.text
    assert out == 'test_log'

    # test for 3 errors by rsync followed by a log output
    test_rsync.side_effect = [error, error, error]
    with pytest.raises(subprocess.CalledProcessError):
        out = download_log(event_config, instance_id, submission_name)
    assert 'Trying to download the log' in caplog.text


@mock.patch('ramp_engine.aws.api._rsync')
def test_rsync_upload_fails(test_rsync):
    test_rsync.side_effect = subprocess.CalledProcessError(255, 'test')
    event_config = read_config(ramp_aws_config_template())['worker']
    instance_id = 0
    submission_name = 'test_submission'
    submissions_dir = 'temp'
    out = upload_submission(event_config, instance_id, submission_name,
                            submissions_dir)
    assert out == 1  # error ocurred and it was caught


@mock.patch('ramp_engine.aws.api._run')
def test_is_spot_terminated_with_CalledProcessError(test_run, caplog):
    test_run.side_effect = subprocess.CalledProcessError(28, 'test')
    event_config = read_config(ramp_aws_config_template())['worker']
    instance_id = 0
    is_spot_terminated(event_config, instance_id)
    assert 'Unable to run curl' in caplog.text


@pytest.mark.parametrize(
    "use_spot_instance",
    [None, True, False]
    )
@mock.patch("boto3.session.Session")
def test_launch_ec2_instances(boto_session_cls, use_spot_instance):
    ''' Check 'use_spot_instance' config with None, True and False'''
    # dummy mock session
    session = boto_session_cls.return_value
    client = session.client.return_value
    describe_images = client.describe_images
    images = {"Images": [{"ImageId": 1, "CreationDate": 123}]}
    describe_images.return_value = images
    event_config = read_config(ramp_aws_config_template())['worker']

    event_config['use_spot_instance'] = use_spot_instance
    launch_ec2_instances(event_config)


@mock.patch('ramp_engine.aws.api.launch_train')
def test_aws_worker_launch_train_error(launch_train, caplog):
    # mock dummy AWS instance
    class DummyInstance:
        id = 1
    launch_train.side_effect = subprocess.CalledProcessError(255, 'test')

    # setup the AWS worker
    event_config = read_config(ramp_aws_config_template())['worker']

    worker = AWSWorker(event_config, submission='starting_kit_local')
    worker.config = event_config
    worker.submission = 'dummy submissions'
    worker.instance = DummyInstance

    # CalledProcessError is thrown inside
    status = worker.launch_submission()
    assert 'test' in caplog.text
    assert 'Cannot start training of submission' in caplog.text
    assert worker.status == 'error'
    assert status == 1


@pytest.mark.parametrize(
    'aws_msg_type, result_none, log_msg',
    [('max_spot', True, 'MaxSpotInstanceCountExceeded'),
     ('unhandled', True, 'this is temporary message'),
     ('correct', False, 'Spot instance request fulfilled')
     ]
)
# set shorter waiting time than in the actual settings
@mock.patch("ramp_engine.aws.api.WAIT_MINUTES", 0.03)
@mock.patch("ramp_engine.aws.api.MAX_TRIES_TO_CONNECT", 4)
@mock.patch("boto3.session.Session")
def test_creating_instances(boto_session_cls, caplog,
                            aws_msg_type, result_none, log_msg):
    ''' test launching more instances than limit on AWS enabled'''
    # info: caplog is a pytest fixture to collect logging info
    # dummy mock session of AWS
    session = boto_session_cls.return_value
    client = session.client.return_value
    describe_images = client.describe_images
    images = {"Images": [{"ImageId": 1, "CreationDate": 123}]}
    describe_images.return_value = images

    error = {
        "ClientError": {
            "Code": "Max spot instance count exceeded"
        }
    }
    event_config = read_config(ramp_aws_config_template())['worker']
    event_config['use_spot_instance'] = True
    request_spot_instances = client.request_spot_instances

    error_max_instances = botocore.exceptions.ClientError(
        error, "MaxSpotInstanceCountExceeded")
    error_unhandled = botocore.exceptions.ParamValidationError(
        report='this is temporary message')
    correct_response = {'SpotInstanceRequests':
                        [{'SpotInstanceRequestId': ['temp']}]
                        }

    if aws_msg_type == 'max_spot':
        aws_response = [error_max_instances, error_max_instances,
                        error_max_instances, error_max_instances]
    elif aws_msg_type == 'unhandled':
        aws_response = [error_unhandled, error_unhandled]
    elif aws_msg_type == 'correct':
        aws_response = [error_max_instances, correct_response]

    request_spot_instances.side_effect = aws_response
    instance, status = launch_ec2_instances(event_config)
    assert (instance is None) == result_none
    assert log_msg in caplog.text


@mock.patch('ramp_engine.aws.api.is_spot_terminated')
@mock.patch('ramp_engine.aws.api.launch_train')
@mock.patch('ramp_engine.aws.api._training_finished')
def test_restart_on_sudden_instance_termination(training_finished,
                                                launch_train, spot_terminated,
                                                caplog):
    class DummyInstance:
        id = 1
    launch_train.return_value = 0

    # setup the AWS worker
    event_config = read_config(ramp_aws_config_template())['worker']

    worker = AWSWorker(event_config, submission='starting_kit_local')
    worker.config = event_config
    worker.submission = 'dummy submissions'
    worker.instance = DummyInstance

    # set the submission did not yet finish training
    training_finished.return_value = False
    spot_terminated.return_value = False

    worker.launch_submission()
    assert worker.status == 'running'
    assert caplog.text == ''

    # call CalledProcessError on checking if submission was finished
    training_finished.side_effect = subprocess.CalledProcessError(255, 'test')
    # make sure that the worker status is set to 'retry'
    assert worker.status == 'retry'
    assert 'Unable to connect to the instance' in caplog.text
    assert 'Adding the submission back to the queue' in caplog.text


def test_aws_worker():
    if not os.path.isfile(os.path.join(HERE, 'config.yml')):
        pytest.skip("Only for local tests for now")

    ramp_kit_dir = os.path.join(HERE, 'kits', 'iris')

    # make sure prediction and log dirs exist, if not, add them
    add_empty_dir(os.path.join(ramp_kit_dir, 'predictions'))
    add_empty_dir(os.path.join(ramp_kit_dir, 'logs'))

    # if the prediction / log files are still there, remove them
    for subdir in os.listdir(os.path.join(ramp_kit_dir, 'predictions')):
        if os.path.isdir(subdir):
            shutil.rmtree(subdir)
    for subdir in os.listdir(os.path.join(ramp_kit_dir, 'logs')):
        if os.path.isdir(subdir):
            shutil.rmtree(subdir)

    worker_config = read_config(os.path.join(HERE, 'config.yml'))['worker']
    worker = AWSWorker(worker_config, submission='starting_kit_local')
    worker.setup()
    assert worker.status == 'setup'
    worker.launch_submission()
    assert worker.status in ('running', 'finished')
    worker.collect_results()
    assert worker.status == 'collected'
    assert os.path.isdir(os.path.join(
        ramp_kit_dir, 'predictions', 'starting_kit_local', 'fold_0'))
    assert os.path.isfile(os.path.join(
        ramp_kit_dir, 'logs', 'starting_kit_local', 'log'))

    worker.teardown()
    assert worker.status == 'killed'


def test_aws_dispatcher(session_toy):  # noqa
    # copy of test_integration_dispatcher but with AWS
    if not os.path.isfile(os.path.join(HERE, 'config.yml')):
        pytest.skip("Only for local tests for now")

    config = read_config(database_config_template())
    event_config = ramp_config_template()
    event_config = read_config(event_config)

    # patch the event_config to match local config.yml for AWS
    aws_event_config = read_config(os.path.join(HERE, 'config.yml'))
    event_config['worker'] = aws_event_config['worker']

    dispatcher = Dispatcher(
        config=config, event_config=event_config, worker=AWSWorker,
        n_workers=-1, hunger_policy='exit'
    )
    dispatcher.launch()

    # the iris kit contain a submission which should fail for each user
    submission = get_submissions(
        session_toy, event_config['ramp']['event_name'], 'training_error'
    )
    assert len(submission) == 2
