"""
For testing the Amazon worker locally, you need:
- the air_passengers starting kit with a 'local_starting_kit' submission
- a config.yml file with credentials to interact with Amazon, and appropriate
  fields for the air_passengers problem:
    - ami_image_name: air_passengers_backend
    - remote_ramp_kit_folder : /home/ubuntu/ramp-kits/air_passengers

"""
import logging
import os
import shutil

import pytest

from rampbkd.aws import AWSWorker
from rampbkd.config import read_backend_config


HERE = os.path.dirname(__file__)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')


def test_aws_worker():
    if not os.path.isfile(os.path.join(HERE, 'config.yml')):
        pytest.skip("Only for local tests for now")

    ramp_kit_dir = os.path.join(HERE, 'kits', 'air_passengers')
    # if the prediction / log files are still there, remove them
    for subdir in os.listdir(os.path.join(ramp_kit_dir, 'predictions')):
        if os.path.isdir(subdir):
            shutil.rmtree(subdir)
    for subdir in os.listdir(os.path.join(ramp_kit_dir, 'logs')):
        if os.path.isdir(subdir):
            shutil.rmtree(subdir)

    conf = read_backend_config(os.path.join(HERE, 'config.yml'))
    worker = AWSWorker(conf['aws'], submission='local_starting_kit',
                       ramp_kit_dir=ramp_kit_dir)
    worker.setup()
    assert worker.status == 'setup'
    worker.launch_submission()
    assert worker.status == 'running'
    worker.collect_results()
    assert worker.status == 'collected'

    assert os.path.isdir(os.path.join(
        ramp_kit_dir, 'predictions', 'local_starting_kit', 'fold_0'))
    assert os.path.isfile(os.path.join(
        ramp_kit_dir, 'logs', 'local_starting_kit', 'log'))

    worker.teardown()
