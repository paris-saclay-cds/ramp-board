
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
    subprocess.check_call(["utils", "--config", path_config_example(),
                           "deploy-ramp-event"])


def teardown_module(module):
    config = read_config(path_config_example())
    shutil.rmtree(config['ramp']['deployment_dir'], ignore_errors=True)
    db, Session = setup_db(config['sqlalchemy'])
    with db.connect() as conn:
        session = Session(bind=conn)
        session.close()
    Model.metadata.drop_all(db)


def test_create_user():
    runner = CliRunner()
    result = runner.invoke(main, ['--config', path_config_example(),
                                  'create-user',
                                  '--login', 'glemaitre',
                                  '--password', 'xxx',
                                  '--lastname', 'xxx',
                                  '--firstname', 'xxx',
                                  '--email', 'xxx',
                                  '--access_level', 'admin'],
                           obj={}, catch_exceptions=False)
    print(result.output)
    assert result.exit_code == 0


def test_approve_user():
    runner = CliRunner()
    result = runner.invoke(main, ['--config', path_config_example(),
                                  'approve-user',
                                  '--login', 'glemaitre'],
                           obj={})
    assert result.exit_code == 0


def test_sign_up_team():
    runner = CliRunner()
    result = runner.invoke(main, ['--config', path_config_example(),
                                  'sign-up-team',
                                  '--name', 'glemaitre'],
                           obj={})
    assert result.exit_code == 0
