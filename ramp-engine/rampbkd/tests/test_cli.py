import shutil

from click.testing import CliRunner

from ramputils import read_config
from ramputils.testing import database_config_template
from ramputils.testing import ramp_config_template

from rampdb.model import Model
from rampdb.utils import setup_db
from rampdb.testing import create_toy_db

from rampbkd.cli import main


def setup_module(module):
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    create_toy_db(database_config, ramp_config)


def teardown_module(module):
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    shutil.rmtree(ramp_config['ramp']['deployment_dir'], ignore_errors=True)
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
                                  "--worker-type", 'CondaEnvWorker',
                                  "--submission", "starting_kit"])
    assert result.exit_code == 0, result.output
