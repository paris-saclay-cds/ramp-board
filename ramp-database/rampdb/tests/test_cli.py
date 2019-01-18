import os
import shutil

from click.testing import CliRunner

from ramputils import read_config
from ramputils import generate_ramp_config
from ramputils.testing import path_config_example

from rampdb.utils import setup_db
from rampdb.model import Model
from rampdb.testing import create_toy_db

from rampdb.cli import main


def setup_module(module):
    config = read_config(path_config_example())
    create_toy_db(config)


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


def test_sign_up_team():
    runner = CliRunner()
    result = runner.invoke(main, ['sign-up-team',
                                  '--config', path_config_example(),
                                  '--event', 'iris_test',
                                  '--team', 'glemaitre'],
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


def test_add_submission():
    ramp_config = generate_ramp_config(path_config_example())
    submission_name = 'new_submission'
    submission_path = os.path.join(ramp_config['ramp_kit_submissions_dir'],
                                   submission_name)
    shutil.copytree(
        os.path.join(ramp_config['ramp_kit_submissions_dir'],
                     'random_forest_10_10'),
        submission_path
    )
    runner = CliRunner()
    result = runner.invoke(main, ['add-submission',
                                  '--config', path_config_example(),
                                  '--event', 'iris_test',
                                  '--team', 'glemaitre',
                                  '--submission', submission_name,
                                  '--path', submission_path],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_get_submission_by_state():
    runner = CliRunner()
    result = runner.invoke(main, ['get-submissions-by-state',
                                  '--config', path_config_example(),
                                  '--event', 'boston_housing_test',
                                  '--state', 'new'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert "ID" in result.output
    assert "name" in result.output
    assert "team" in result.output
    assert "path" in result.output
    assert "state" in result.output
    result = runner.invoke(main, ['get-submissions-by-state',
                                  '--config', path_config_example(),
                                  '--event', 'iris_test',
                                  '--state', 'scored'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert 'No submission for this event and this state' in result.output


def test_set_submission_state():
    runner = CliRunner()
    result = runner.invoke(main, ['set-submission-state',
                                  '--config', path_config_example(),
                                  '--submission-id', 3,
                                  '--state', 'scored'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_update_leaderboards():
    runner = CliRunner()
    result = runner.invoke(main, ['update-leaderboards',
                                  '--config', path_config_example(),
                                  '--event', 'iris_test'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_update_user_leaderboards():
    runner = CliRunner()
    result = runner.invoke(main, ['update-user-leaderboards',
                                  '--config', path_config_example(),
                                  '--event', 'iris_test',
                                  '--user', 'glemaitre'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output


def test_update_all_user_leaderboards():
    runner = CliRunner()
    result = runner.invoke(main, ['update-all-users-leaderboards',
                                  '--config', path_config_example(),
                                  '--event', 'iris_test'],
                           catch_exceptions=False)
    assert result.exit_code == 0, result.output
