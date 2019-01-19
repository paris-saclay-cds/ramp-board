import shutil
import subprocess

from click.testing import CliRunner

from ramputils import read_config
from ramputils.testing import path_config_example

from rampdb.model import Model
from rampdb.utils import setup_db
from rampdb.testing import create_toy_db

from rampbkd.cli import main


def setup_module(module):
    config = read_config(path_config_example())
    create_toy_db(config)


def teardown_module(module):
    config = read_config(path_config_example())
    shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
    db, _ = setup_db(config['sqlalchemy'])
    Model.metadata.drop_all(db)


def test_dispatcher():
    runner = CliRunner()
    result = runner.invoke(main, ["dispatcher",
                                  "--config", path_config_example(),
                                  "--verbose"])
    assert result.exit_code == 0, result.output
