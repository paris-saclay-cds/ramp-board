import shutil

from click.testing import CliRunner

from ramp_utils import read_config
from ramp_utils.testing import database_config_template
from ramp_utils.testing import ramp_config_template

from ramp_database.model import Model
from ramp_database.utils import setup_db
from ramp_database.testing import create_toy_db

from ramp_engine.cli import main


def setup_module(module):
    database_config = read_config(database_config_template())
    ramp_config = ramp_config_template()
    module.deployment_dir = create_toy_db(database_config, ramp_config)


def teardown_module(module):
    database_config = read_config(database_config_template())
    shutil.rmtree(module.deployment_dir, ignore_errors=True)
    db, _ = setup_db(database_config['sqlalchemy'])
    Model.metadata.drop_all(db)


def test_dispatcher():
    runner = CliRunner()
    result = runner.invoke(main, ["dispatcher",
                                  "--config", database_config_template(),
                                  "--event-config", ramp_config_template()])
    assert result.exit_code == 0, result.output


def test_worker():
    runner = CliRunner()
    result = runner.invoke(main, ["worker",
                                  "--config", ramp_config_template(),
                                  "--event-config", ramp_config_template(),
                                  "--submission", "starting_kit"])
    assert result.exit_code == 0, result.output
