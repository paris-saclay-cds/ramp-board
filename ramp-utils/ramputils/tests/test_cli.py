import shutil

from click.testing import CliRunner

from rampdb.utils import setup_db
from rampdb.model import Model

from ramputils import read_config
from ramputils.testing import path_config_example

from ramputils.cli import main


def teardown_module(module):
    config = read_config(path_config_example())
    shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
    db, Session = setup_db(config['sqlalchemy'])
    Model.metadata.drop_all(db)


def test_deploy_ramp_event():
    runner = CliRunner()
    result = runner.invoke(main, ['deploy-ramp-event',
                                  '--config', path_config_example()])
    assert result.exit_code == 0
