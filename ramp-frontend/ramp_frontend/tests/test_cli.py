import shutil
import subprocess
import time

from ramp_utils import read_config
from ramp_utils import generate_ramp_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.utils import setup_db
from ramp_database.model import Model
from ramp_database.testing import create_toy_db

from ramp_frontend.cli import main


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
    try:
        proc = subprocess.Popen(["ramp", "frontend", "test-launch",
                                "--config", database_config_template()],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    finally:
        # wait couple of seconds for the serve to start
        time.sleep(5)
        proc.terminate()
        assert b'Serving Flask app "ramp-frontend"' in proc.stdout.read()
