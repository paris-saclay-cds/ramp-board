"""
For testing the Amazon worker locally, you need to:
copy the _config.yml file and rename it config.yml, then update it using the
credentials to interact with Amazon

"""
import botocore
import logging
import os
import shutil
import subprocess

import pytest

from ramp_database.tools.submission import get_submissions
from ramp_engine.aws.api import is_spot_terminated, launch_ec2_instances
from ramp_engine import Dispatcher, AWSWorker
from ramp_utils import generate_worker_config, read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

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


@mock.patch('ramp_engine.aws.api._run')
def test_is_spot_terminated_with_CalledProcessError(test_run, caplog):
    test_run.side_effect = subprocess.CalledProcessError(28, 'test')
    config = read_config(os.path.join(HERE, '_config.yml'))
    instance_id = 0
    is_spot_terminated(config, instance_id)
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
    images = {"Images": [{"ImageId": 1}]}
    describe_images.return_value = images
    config = read_config(os.path.join(HERE, '_config.yml'))

    config['worker']['use_spot_instance'] = use_spot_instance
    launch_ec2_instances(config['worker'])


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
    images = {"Images": [{"ImageId": 1}]}
    describe_images.return_value = images

    error = {
        "ClientError": {
            "Code": "Max spot instance count exceeded"
        }
    }
    config = read_config(os.path.join(HERE, '_config.yml'))
    config['worker']['use_spot_instance'] = True
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
    instance, = launch_ec2_instances(config['worker'])
    assert (instance is None) == result_none
    assert log_msg in caplog.text


def test_aws_worker():
    if not os.path.isfile(os.path.join(HERE, 'config.yml')):
        pytest.skip("Only for local tests for now")

    ramp_kit_dir = os.path.join(HERE, 'kits', 'iris')

    # make sure predictio and log dirs exist, if not, add them
    add_empty_dir(os.path.join(ramp_kit_dir, 'predictions'))
    add_empty_dir(os.path.join(ramp_kit_dir, 'logs'))

    # if the prediction / log files are still there, remove them
    for subdir in os.listdir(os.path.join(ramp_kit_dir, 'predictions')):
        if os.path.isdir(subdir):
            shutil.rmtree(subdir)
    for subdir in os.listdir(os.path.join(ramp_kit_dir, 'logs')):
        if os.path.isdir(subdir):
            shutil.rmtree(subdir)

    config = read_config(os.path.join(HERE, 'config.yml'))
    worker_config = generate_worker_config(config)
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
