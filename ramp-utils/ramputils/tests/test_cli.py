import shutil

from click.testing import CliRunner

from rampdb.utils import setup_db
from rampdb.model import Model

from ramputils import read_config
from ramputils.testing import database_config_template
from ramputils.testing import ramp_config_template

from ramputils.cli import main


def teardown_function(function):
    database_config = read_config(database_config_template())
    ramp_config = read_config(ramp_config_template())
    shutil.rmtree(ramp_config['ramp']['deployment_dir'], ignore_errors=True)
    db, _ = setup_db(database_config['sqlalchemy'])
    Model.metadata.drop_all(db)


def test_deploy_ramp_event():
    runner = CliRunner()
    result = runner.invoke(main, ['deploy-ramp-event',
                                  '--config', database_config_template(),
                                  '--event-config', ramp_config_template()])
    assert result.exit_code == 0, result.output
    result = runner.invoke(main, ['deploy-ramp-event',
                                  '--config', database_config_template(),
                                  '--event-config', ramp_config_template(),
                                  '--force'])
    assert result.exit_code == 0, result.output
