
import shutil
import subprocess

from click.testing import CliRunner

from ramputils import read_config
from ramputils import generate_ramp_config
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
                                  '--access-level', 'admin'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_approve_user():
    runner = CliRunner()
    result = runner.invoke(main, ['approve-user',
                                  '--config', path_config_example(),
                                  '--login', 'glemaitre'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_sign_up_team():
    runner = CliRunner()
    result = runner.invoke(main, ['sign-up-team',
                                  '--config', path_config_example(),
                                  '--event', 'iris_test',
                                  '--team', 'glemaitre'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_add_problem():
    runner = CliRunner()
    ramp_config = generate_ramp_config(path_config_example())
    result = runner.invoke(main, ['add-problem',
                                  '--config', path_config_example(),
                                  '--problem', 'iris',
                                  '--kits-dir', ramp_config['ramp_kits_dir'],
                                  '--data-dir', ramp_config['ramp_data_dir'],
                                  '--force', True],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_add_event():
    runner = CliRunner()
    ramp_config = generate_ramp_config(path_config_example())
    result = runner.invoke(main, ['add-event',
                                  '--config', path_config_example(),
                                  '--problem', 'iris',
                                  '--event', 'iris_test',
                                  '--title', 'Iris classification',
                                  '--sandbox', ramp_config['sandbox_name'],
                                  '--submissions-dir',
                                  ramp_config['ramp_submissions_dir'],
                                  '--is-public', False,
                                  '--force', True],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_add_event_admin():
    runner = CliRunner()
    result = runner.invoke(main, ['add-event-admin',
                                  '--config', path_config_example(),
                                  '--event', 'iris_test',
                                  '--user', 'glemaitre'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output
