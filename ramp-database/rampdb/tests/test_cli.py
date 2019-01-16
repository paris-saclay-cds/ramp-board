
import shutil
import subprocess

from click.testing import CliRunner

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.utils import setup_db
from rampdb.model import Model
from rampdb.testing import create_test_db

from rampdb.cli import main


def setup_module(module):
    config = read_config(path_config_example())
    create_test_db(config)
    subprocess.check_call(["ramp-utils",
                           "deploy-ramp-event",
                           "--config", path_config_example()])


def teardown_module(module):
    config = read_config(path_config_example())
    shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
    db, Session = setup_db(config['sqlalchemy'])
    Model.metadata.drop_all(db)


def test_add_user():
    runner = CliRunner()
    result = runner.invoke(main, ['add-user',
                                  '--config', path_config_example(),
                                  '--login', 'glemaitre',
                                  '--password', 'xxx',
                                  '--lastname', 'xxx',
                                  '--firstname', 'xxx',
                                  '--email', 'xxx',
                                  '--access_level', 'admin'],
                           catch_exceptions=False)
    assert result.exit_code == 0


def test_approve_user():
    runner = CliRunner()
    result = runner.invoke(main, ['approve-user',
                                  '--config', path_config_example(),
                                  '--login', 'glemaitre'])
    assert result.exit_code == 0


def test_sign_up_team():
    runner = CliRunner()
    result = runner.invoke(main, ['sign-up-team',
                                  '--config', path_config_example(),
                                  '--name', 'glemaitre'])
    assert result.exit_code == 0
