import shutil

import pytest

from click.testing import CliRunner

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Model
from ramp_database.utils import setup_db
from ramp_database.testing import create_toy_db

from ramp_engine.cli import main


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


@pytest.mark.parametrize(
    "verbose_params", [None, "--verbose", "-vv"]
)
def test_dispatcher(verbose_params, make_toy_db):
    runner = CliRunner()
    cmd = ["dispatcher",
           "--config", database_config_template(),
           "--event-config", ramp_config_template()]
    if verbose_params is not None:
        cmd += [verbose_params]
    result = runner.invoke(main, cmd)
    assert result.exit_code == 0, result.output


@pytest.mark.parametrize(
    "verbose_params", [None, "--verbose", "-vv"]
)
def test_worker(verbose_params, make_toy_db):
    runner = CliRunner()
    cmd = ["worker",
           "--event-config", ramp_config_template(),
           "--submission", "starting_kit"]
    if verbose_params is not None:
        cmd += [verbose_params]
    result = runner.invoke(main, cmd)
    assert result.exit_code == 0, result.output
