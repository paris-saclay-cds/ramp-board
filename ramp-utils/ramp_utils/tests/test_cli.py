import os
import shutil

from click.testing import CliRunner

from ramp_database.utils import setup_db
from ramp_database.model import Model

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_utils.cli import main


def teardown_function(function):
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    shutil.rmtree(ramp_config['ramp']['deployment_dir'], ignore_errors=True)
    db, _ = setup_db(database_config['sqlalchemy'])
    Model.metadata.drop_all(db)


def test_setup_init():
    ramp_config = read_config(ramp_config_template())
    deployment_dir = ramp_config['ramp']['deployment_dir']
    os.mkdir(deployment_dir)
    runner = CliRunner()
    result = runner.invoke(main, ['init',
                                  '--deployment-dir', deployment_dir])
    assert result.exit_code == 0, result.output


def test_setup_init_event():
    ramp_config = read_config(ramp_config_template())
    deployment_dir = ramp_config['ramp']['deployment_dir']
    os.mkdir(deployment_dir)
    runner = CliRunner()
    result = runner.invoke(main, ['init-event',
                                  '--name', 'iris_test',
                                  '--deployment-dir', deployment_dir])
    assert result.exit_code == 0, result.output


def test_deploy_ramp_event():
    runner = CliRunner()
    result = runner.invoke(main, ['deploy-event',
                                  '--config', database_config_template(),
                                  '--event-config', ramp_config_template()])
    assert result.exit_code == 0, result.output
    result = runner.invoke(main, ['deploy-event',
                                  '--config', database_config_template(),
                                  '--event-config', ramp_config_template(),
                                  '--force'])
    assert result.exit_code == 0, result.output
