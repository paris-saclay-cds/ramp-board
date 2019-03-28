import os
import shutil
import subprocess
import sys
import time

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.utils import setup_db
from ramp_database.model import Model
from ramp_database.testing import create_toy_db


def setup_module(module):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    module.deployment_dir = create_toy_db(database_config, ramp_config)


def teardown_module(module):
    database_config = read_config(database_config_template())
    shutil.rmtree(module.deployment_dir, ignore_errors=True)
    db, _ = setup_db(database_config['sqlalchemy'])
    Model.metadata.drop_all(db)


def test_test_launch():
    # pass environment to subprocess
    cmd = [sys.executable, '-m', 'coverage', 'run', '-m']
    cmd += ["ramp_frontend.cli", "test-launch",
            "--config", database_config_template()]
    try:
        proc = subprocess.Popen(cmd,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                env=os.environ.copy())
        stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        pass
    finally:
        proc.terminate()
        print(proc.stdout.read())
