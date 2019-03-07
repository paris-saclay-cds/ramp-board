import os
import shutil

import pytest

from click.testing import CliRunner

from ramp_database.utils import setup_db
from ramp_database.model import Model

from ramp_utils import read_config
from ramp_utils.utils import commonpath
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_utils.cli import main


@pytest.fixture
def deployment_dir():
    ramp_config = read_config(ramp_config_template())
    return commonpath(
        [ramp_config['ramp']['kit_dir'], ramp_config['ramp']['data_dir']]
    )


def setup_function(function):
    ramp_config = read_config(ramp_config_template())
    function.deployment_dir = commonpath(
        [ramp_config['ramp']['kit_dir'], ramp_config['ramp']['data_dir']]
    )


def teardown_function(function):
    database_config = read_config(database_config_template())
    # FIXME: we are recreating the deployment directory but it should be
    # replaced by an temporary creation of folder.
    shutil.rmtree(function.deployment_dir, ignore_errors=True)
    db, _ = setup_db(database_config['sqlalchemy'])
    Model.metadata.drop_all(db)


def test_setup_init(deployment_dir):
    try:
        os.mkdir(deployment_dir)
        runner = CliRunner()
        result = runner.invoke(main, ['init',
                                      '--deployment-dir', deployment_dir])
        assert result.exit_code == 0, result.output
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)


def test_setup_init_event(deployment_dir):
    try:
        os.mkdir(deployment_dir)
        runner = CliRunner()
        result = runner.invoke(main, ['init-event',
                                      '--name', 'iris_test',
                                      '--deployment-dir', deployment_dir])
        assert result.exit_code == 0, result.output
    finally:
        shutil.rmtree(deployment_dir, ignore_errors=True)


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
