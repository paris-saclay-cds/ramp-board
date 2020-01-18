import os
import shutil
import signal
import subprocess
import time

import pytest

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.utils import setup_db
from ramp_database.model import Model
from ramp_database.testing import create_toy_db


@pytest.fixture(scope="module")
def make_toy_db(database_connection):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    try:
        deployment_dir = create_toy_db(database_config, ramp_config)
        yield
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)
        db, _ = setup_db(database_config['sqlalchemy'])
        Model.metadata.drop_all(db)


def test_test_launch(make_toy_db):
    # pass environment to subprocess
    cmd = ['python', '-m']
    cmd += ["ramp_frontend.cli", "test-launch",
            "--config", database_config_template()]
    proc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        env=os.environ.copy())
    # wait for 5 seconds before to terminate the server
    time.sleep(5)
    proc.send_signal(signal.SIGINT)
    stdout, _ = proc.communicate()
    assert b'Serving Flask app "ramp-frontend"' in stdout
