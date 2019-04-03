"""
For testing the Amazon worker locally, you need:
- the iris starting kit with a 'starting_kit_local' submission
- a config.yml file with credentials to interact with Amazon, and appropriate
  fields for the iris problem:
    - ami_image_name: ramp_aws_iris_test
    - remote_ramp_kit_folder : /home/ubuntu/ramp-kits/iris

"""
import logging
import os
import shutil

import pytest

from ramp_database.tools.submission import get_submissions
from ramp_engine import Dispatcher, AWSWorker
from ramp_utils import generate_worker_config, read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from .test_dispatcher import session_toy  # noqa


HERE = os.path.dirname(__file__)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def test_aws_worker():
    if not os.path.isfile(os.path.join(HERE, 'config.yml')):
        pytest.skip("Only for local tests for now")

    ramp_kit_dir = os.path.join(HERE, 'kits', 'iris')
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
